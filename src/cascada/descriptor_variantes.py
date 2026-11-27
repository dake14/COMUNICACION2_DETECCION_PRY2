"""Variantes del descriptor de imágenes: más ancho y con cabeza adversaria de dominio.

CONTRATO (unidad U6, cerrado):

Dependencias permitidas (lista cerrada): torch, torch.nn, torch.nn.functional, src.cascada.descriptor.Descriptor.

- ``class RamaAncha(nn.Module)`` con ``__init__(self, relleno)``: igual que ``src.cascada.descriptor.Rama`` pero con
  capas (salida, núcleo) = (32, 5), (64, 3), (128, 3), (192, 3): cada una ``Conv2d(entrada, salida, nucleo, stride=2,
  padding=nucleo // 2, padding_mode=relleno)``, ``GroupNorm(8, salida)``, ``SiLU()``; entrada inicial 1 canal.
  ``forward(x)``: ``h = self.red(x)``; devuelve ``torch.cat((h.mean((2, 3)), h.amax((2, 3))), 1)`` (384 valores).
- ``class DescriptorAncho(nn.Module)`` con ``__init__(self, dimension=128)``: ``self.cart = RamaAncha("zeros")``,
  ``self.polar = RamaAncha("circular")``, ``self.oculta = nn.Sequential(Linear(768, 128), LayerNorm(128), SiLU())``,
  ``self.huella = Linear(128, dimension)``, ``self.geometria = Linear(128, 2)``.
  ``forward(x)`` con x (B, 2, 64, 64): ``h = self.oculta(torch.cat((self.cart(x[:, :1]), self.polar(x[:, 1:])), 1))``;
  devuelve ``(F.normalize(self.huella(h), dim=1), self.geometria(h))``.
- ``class _Inversion(torch.autograd.Function)``: ``forward(ctx, x, alfa)`` guarda alfa y devuelve ``x.view_as(x)``;
  ``backward(ctx, g)`` devuelve ``(-ctx.alfa * g, None)``.
- ``invertir_gradiente(x, alfa) -> torch.Tensor``: ``_Inversion.apply(x, float(alfa))``.
- ``class DescriptorAdversario(nn.Module)`` con ``__init__(self, n_dominios, dimension=128)``:
  ``self.base = Descriptor(dimension)``; ``self.dominio = nn.Sequential(Linear(128, 64), SiLU(), Linear(64, n_dominios))``.
  ``forward(x, alfa=0.0)``: ``b = self.base``;
  ``h = b.oculta(torch.cat((b.cart(x[:, :1]), b.polar(x[:, 1:])), 1))``;
  devuelve ``(F.normalize(b.huella(h), dim=1), b.geometria(h), self.dominio(invertir_gradiente(h, alfa)))``.
  ``descriptor(self)`` devuelve ``self.base``.

Criterio de aceptación: con entrada ``torch.rand(4, 2, 64, 64)``, ``DescriptorAncho`` da (4, 128) de norma 1 y (4, 2);
``DescriptorAdversario(5)`` da (4, 128), (4, 2), (4, 5); el gradiente que llega a ``h`` desde la cabeza de dominio
tiene signo opuesto y escala ``alfa`` respecto al de la cabeza sola (comprobable con ``invertir_gradiente`` en un
tensor con ``requires_grad``); ``descriptor()`` devuelve un ``Descriptor`` cuya salida coincide con las dos primeras.
"""
import torch
import torch.nn.functional as F
from torch import nn

from src.cascada.descriptor import Descriptor


class RamaAncha(nn.Module):
    def __init__(self, relleno):
        super().__init__()
        capas, entrada = [], 1
        for salida, nucleo in ((32, 5), (64, 3), (128, 3), (192, 3)):
            capas += [
                nn.Conv2d(entrada, salida, nucleo, stride=2, padding=nucleo // 2, padding_mode=relleno),
                nn.GroupNorm(8, salida),
                nn.SiLU(),
            ]
            entrada = salida
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        h = self.red(x)
        return torch.cat((h.mean((2, 3)), h.amax((2, 3))), 1)


class DescriptorAncho(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.cart = RamaAncha("zeros")
        self.polar = RamaAncha("circular")
        self.oculta = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.SiLU())
        self.huella = nn.Linear(128, dimension)
        self.geometria = nn.Linear(128, 2)

    def forward(self, x):
        h = self.oculta(torch.cat((self.cart(x[:, :1]), self.polar(x[:, 1:])), 1))
        return F.normalize(self.huella(h), dim=1), self.geometria(h)


class _Inversion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alfa):
        ctx.alfa = alfa
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alfa * g, None


def invertir_gradiente(x, alfa):
    return _Inversion.apply(x, float(alfa))


class DescriptorAdversario(nn.Module):
    def __init__(self, n_dominios, dimension=128):
        super().__init__()
        self.base = Descriptor(dimension)
        self.dominio = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, n_dominios))

    def forward(self, x, alfa=0.0):
        b = self.base
        h = b.oculta(torch.cat((b.cart(x[:, :1]), b.polar(x[:, 1:])), 1))
        return F.normalize(b.huella(h), dim=1), b.geometria(h), self.dominio(invertir_gradiente(h, alfa))

    def descriptor(self):
        return self.base
