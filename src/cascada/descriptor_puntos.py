"""Descriptor sobre los símbolos como conjunto de puntos (DeepSets / PointNet).

CONTRATO (unidad U5, cerrado):

Dependencias permitidas (lista cerrada): numpy, torch, torch.nn, torch.nn.functional.

- ``PUNTOS = 512``
- ``entrada_puntos(simbolos, n=PUNTOS, rng=None) -> np.ndarray`` float32 de forma (3, n):
  ``z = np.asarray(simbolos, complex128).ravel()``; si ``len(z) == 0`` devolver ceros (3, n).
  ``z = z - z.mean()``; ``z = z / sqrt(max(mean(|z|^2), 1e-12))``.
  Índices: si ``len(z) >= n``: ``rng.choice(len(z), n, replace=False)`` cuando hay rng, si no
  ``np.linspace(0, len(z) - 1, n).astype(int)``; si ``len(z) < n``: ``np.resize(np.arange(len(z)), n)``.
  Filas: parte real, parte imaginaria y módulo de los puntos elegidos.
- ``aumentar_puntos(x) -> torch.Tensor``: ``x`` de forma (B, 3, N). Por ejemplo, ángulo U(0, 2pi) y conjugación con
  probabilidad 0.5 (ambos con ``torch.rand`` en el dispositivo de ``x``): primero conjugar (negar la fila 1),
  luego rotar las filas 0 y 1 por el ángulo; la fila 2 no cambia. Devuelve un tensor nuevo, sin modificar ``x``.
- ``class DescriptorPuntos(nn.Module)`` con ``__init__(self, dimension=128)``:
  ``self.local = nn.Sequential(Conv1d(3, 64, 1), GroupNorm(8, 64), SiLU(), Conv1d(64, 128, 1), GroupNorm(8, 128), SiLU(),
  Conv1d(128, 256, 1), GroupNorm(8, 256), SiLU())``;
  ``self.oculta = nn.Sequential(Linear(512, 128), LayerNorm(128), SiLU())``;
  ``self.huella = Linear(128, dimension)``; ``self.geometria = Linear(128, 2)``.
  ``forward(x)``: ``h = self.local(x)``; ``g = torch.cat((h.mean(2), h.amax(2)), 1)``; ``h = self.oculta(g)``;
  devuelve ``(F.normalize(self.huella(h), dim=1), self.geometria(h))``.

Criterio de aceptación: ``DescriptorPuntos()(torch.randn(4, 3, 512))`` da huellas (4, 128) de norma 1 y geometría
(4, 2); la salida no cambia al permutar los puntos (tolerancia 1e-5 en modo eval); ``entrada_puntos`` con 100 o con
5000 símbolos da forma (3, 512) y es determinista sin rng; ``aumentar_puntos`` conserva la fila 2 y el módulo.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PUNTOS = 512


def entrada_puntos(simbolos, n=PUNTOS, rng=None):
    z = np.asarray(simbolos, np.complex128).ravel()
    if len(z) == 0:
        return np.zeros((3, n), dtype=np.float32)
    z = z - z.mean()
    z = z / np.sqrt(max(float(np.mean(np.abs(z) ** 2)), 1e-12))
    if len(z) >= n:
        if rng is not None:
            idx = rng.choice(len(z), n, replace=False)
        else:
            idx = np.linspace(0, len(z) - 1, n).astype(int)
    else:
        idx = np.resize(np.arange(len(z)), n)
    p = z[idx]
    return np.stack((p.real, p.imag, np.abs(p)), axis=0).astype(np.float32)


def aumentar_puntos(x):
    b = x.shape[0]
    angulo = torch.rand(b, 1, device=x.device, dtype=x.dtype) * (2.0 * np.pi)
    conj = torch.rand(b, 1, device=x.device) < 0.5
    re = x[:, 0]
    im = torch.where(conj, -x[:, 1], x[:, 1])
    c = torch.cos(angulo)
    s = torch.sin(angulo)
    re_rot = c * re - s * im
    im_rot = s * re + c * im
    return torch.stack((re_rot, im_rot, x[:, 2]), dim=1)


class DescriptorPuntos(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 256, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        self.oculta = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.huella = nn.Linear(128, dimension)
        self.geometria = nn.Linear(128, 2)

    def forward(self, x):
        h = self.local(x)
        g = torch.cat((h.mean(2), h.amax(2)), 1)
        h = self.oculta(g)
        return F.normalize(self.huella(h), dim=1), self.geometria(h)
