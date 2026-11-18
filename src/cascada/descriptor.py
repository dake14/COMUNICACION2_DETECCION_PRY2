"""Etapa 4: CNN que resume la densidad de la constelación sin usar clases."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BINS = 64
LIMITE = 2.2


def imagen_densidad(simbolos, bins=BINS):
    """Devuelve (2, bins, bins): densidad cartesiana y polar en escala logarítmica."""
    z = np.asarray(simbolos, dtype=np.complex128)
    z = z - z.mean()
    z = z / np.sqrt(max(np.mean(np.abs(z) ** 2), 1e-12))
    cart, _, _ = np.histogram2d(z.imag, z.real, bins=bins, range=[[-LIMITE, LIMITE], [-LIMITE, LIMITE]])
    polar, _, _ = np.histogram2d(np.abs(z), np.angle(z), bins=bins, range=[[0, LIMITE], [-np.pi, np.pi]])
    escala = bins * bins / (4.0 * max(len(z), 1))
    imagen = np.stack((cart, polar)).astype(np.float32) * escala
    return np.clip(np.log1p(imagen) / 4.0, 0.0, 1.5).astype(np.float32)


def aumentar(x):
    """Rotación global y conjugación aleatorias, aplicadas en el espacio de imagen."""
    n = len(x)
    angulo = 2 * np.pi * torch.rand(n)
    cos, sen = angulo.cos(), angulo.sin()
    matriz = torch.stack((torch.stack((cos, -sen, 0 * cos), 1), torch.stack((sen, cos, 0 * cos), 1)), 1)
    malla = F.affine_grid(matriz, (n, 1, BINS, BINS), align_corners=False)
    cart = F.grid_sample(x[:, :1], malla, align_corners=False, padding_mode="zeros")
    polar = x[:, 1:].clone()
    desplazamientos = (angulo / (2 * np.pi) * BINS).long()
    for i, d in enumerate(desplazamientos.tolist()):
        polar[i] = torch.roll(polar[i], d, dims=-1)
    salida = torch.cat((cart, polar), 1)
    voltear = torch.rand(n) < 0.5
    if voltear.any():
        v = salida[voltear]
        salida[voltear] = torch.cat((torch.flip(v[:, :1], (-2,)), torch.flip(v[:, 1:], (-1,))), 1)
    return salida


class Rama(nn.Module):
    def __init__(self, relleno):
        super().__init__()
        capas, entrada = [], 1
        for salida, nucleo in ((16, 5), (32, 3), (64, 3), (96, 3)):
            capas += [nn.Conv2d(entrada, salida, nucleo, stride=2, padding=nucleo // 2, padding_mode=relleno),
                      nn.GroupNorm(8, salida), nn.SiLU()]
            entrada = salida
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        h = self.red(x)
        return torch.cat((h.mean((2, 3)), h.amax((2, 3))), 1)


class Descriptor(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.cart = Rama("zeros")
        self.polar = Rama("circular")
        self.oculta = nn.Sequential(nn.Linear(384, 128), nn.LayerNorm(128), nn.SiLU())
        self.huella = nn.Linear(128, dimension)
        self.geometria = nn.Linear(128, 2)

    def forward(self, x):
        h = self.oculta(torch.cat((self.cart(x[:, :1]), self.polar(x[:, 1:])), 1))
        return F.normalize(self.huella(h), dim=1), self.geometria(h)


def perdida_contrastiva(z1, z2, temperatura=0.1):
    z = torch.cat((z1, z2))
    similitud = z @ z.T / temperatura
    similitud.fill_diagonal_(-1e4)
    n = len(z1)
    objetivo = torch.cat((torch.arange(n, 2 * n), torch.arange(n)))
    return F.cross_entropy(similitud, objetivo)


@torch.inference_mode()
def calcular_huellas(modelo, imagenes, lote=256):
    modelo.eval()
    x = torch.as_tensor(np.asarray(imagenes, dtype=np.float32))
    z, g = [], []
    for i in range(0, len(x), lote):
        a, b = modelo(x[i:i + lote])
        z.append(a)
        g.append(b)
    return torch.cat(z).numpy(), torch.cat(g).numpy()


def guardar_descriptor(modelo, ruta, extra=None):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    torch.save({"estado": modelo.state_dict(), "extra": extra or {}}, temporal)
    temporal.replace(ruta)


def cargar_descriptor(ruta):
    modelo = Descriptor()
    modelo.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True)["estado"])
    return modelo.eval()
