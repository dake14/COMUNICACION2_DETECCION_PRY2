"""Etapa 2: red convolucional dilatada que limpia canal y ruido."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MAX_MUESTRAS = 32768
BORDE = 32


class Bloque(nn.Module):
    def __init__(self, canales, dilatacion):
        super().__init__()
        self.norma = nn.GroupNorm(8, canales)
        self.conv = nn.Conv1d(canales, canales, 5, padding=2 * dilatacion, dilation=dilatacion)
        self.mezcla = nn.Conv1d(canales, canales, 1)

    def forward(self, x):
        return x + self.mezcla(F.silu(self.conv(self.norma(x))))


class Recuperador(nn.Module):
    def __init__(self, canales=48, pilas=2):
        super().__init__()
        self.canales = canales
        self.pilas = pilas
        self.entrada = nn.Conv1d(2, canales, 7, padding=3)
        self.cuerpo = nn.Sequential(*[Bloque(canales, 2 ** i) for _ in range(pilas) for i in range(6)])
        self.salida = nn.Conv1d(canales, 2, 1)

    def forward(self, x):
        return x + self.salida(self.cuerpo(F.silu(self.entrada(x))))


def a_tensor(z):
    z = np.atleast_2d(z)
    return torch.from_numpy(np.stack((z.real, z.imag), axis=1).astype(np.float32))


def perdida_fase_invariante(y, t, borde=BORDE):
    """Error cuadrático relativo tras alinear una fase global por ejemplo."""
    y, t = y[..., borde:-borde], t[..., borde:-borde]
    yc = torch.complex(y[:, 0], y[:, 1])
    tc = torch.complex(t[:, 0], t[:, 1])
    producto = torch.sum(yc * tc.conj(), dim=-1)
    giro = (producto.conj() / (producto.abs() + 1e-9)).detach().unsqueeze(-1)
    error = torch.mean(torch.abs(yc * giro - tc) ** 2, dim=-1)
    return torch.mean(error / (torch.mean(torch.abs(tc) ** 2, dim=-1) + 1e-9))


@torch.inference_mode()
def aplicar_recuperador(modelo, iq8):
    z = np.asarray(iq8, dtype=np.complex64)[:MAX_MUESTRAS]
    z = z / np.sqrt(max(np.mean(np.abs(z) ** 2), 1e-12))
    modelo.eval()
    y = modelo(a_tensor(z))[0].numpy()
    return y[0] + 1j * y[1]


def guardar_recuperador(modelo, ruta, extra=None):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    torch.save({"estado": modelo.state_dict(), "canales": modelo.canales, "pilas": modelo.pilas,
                "extra": extra or {}}, temporal)
    temporal.replace(ruta)


def cargar_recuperador(ruta):
    d = torch.load(ruta, map_location="cpu", weights_only=True)
    modelo = Recuperador(d["canales"], d["pilas"])
    modelo.load_state_dict(d["estado"])
    return modelo.eval()
