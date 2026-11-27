"""Normalización por ancho de banda y las cuatro vistas de los especialistas, sin conocer la tasa de símbolo."""
# ─────────────────────────────  Imports  ─────────────────────────────
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly, welch

# ─────────────────────────────  Config  ──────────────────────────────
OCUPACION = 0.5
FACTOR_MIN, FACTOR_MAX = 1 / 64, 2.0
LARGO_MAX = 16384
LARGO_MIN = 64
NPERSEG_BANDA = 1024
BINS_ESPECTRO = 256
POTENCIAS = (2, 4, 8)
BINS_ENVOLVENTE = 64
RETARDOS = 64
BINS_FRECUENCIA = 128
SUAVIZADO = 4
FORMAS = {"espectro": (1, BINS_ESPECTRO), "potencias": (len(POTENCIAS), BINS_ESPECTRO),
          "envolvente": (2, BINS_ENVOLVENTE), "frecuencia": (2, BINS_FRECUENCIA)}
VISTAS = tuple(FORMAS)


# ─────────────────────────────  Banda  ───────────────────────────────
def _psd(x, nperseg):
    f, p = welch(x, nperseg=nperseg, return_onesided=False, detrend=False)
    orden = np.argsort(f)
    return f[orden], p[orden]


def estimar_banda(x):
    """Centro y ancho (ciclos por muestra) del 95 % de la potencia sobre el piso de ruido."""
    f, p = _psd(x, min(NPERSEG_BANDA, max(16, len(x) // 4)))
    exceso = np.maximum(p - np.percentile(p, 10), 0.0)
    if exceso.sum() <= 0:
        return 0.0, OCUPACION
    centro = float(np.angle(np.sum(exceso * np.exp(2j * np.pi * f))) / (2 * np.pi))
    relativa = (f - centro + 0.5) % 1.0 - 0.5
    orden = np.argsort(relativa)
    relativa, exceso = relativa[orden], exceso[orden]
    acumulada = np.cumsum(exceso) / exceso.sum()
    bajo = relativa[min(np.searchsorted(acumulada, 0.025), len(relativa) - 1)]
    alto = relativa[min(np.searchsorted(acumulada, 0.975), len(relativa) - 1)]
    return float(centro + (bajo + alto) / 2), float(max(alto - bajo, 2.0 / len(f)))


def normalizar_banda(x):
    """Lleva el centro del espectro a cero y remuestrea para que el ancho ocupado sea la mitad de la tasa."""
    x = np.asarray(x, dtype=np.complex128).ravel()
    centro, ancho = estimar_banda(x)
    x = x * np.exp(-2j * np.pi * centro * np.arange(len(x)))
    factor = Fraction(float(np.clip(ancho / OCUPACION, FACTOR_MIN, FACTOR_MAX))).limit_denominator(64)
    if factor != 1:
        x = resample_poly(x, factor.numerator, factor.denominator)
    x = x[:LARGO_MAX]
    x = x / np.sqrt(max(np.mean(np.abs(x) ** 2), 1e-30))
    return x, {"centro": centro, "ancho": ancho, "factor": float(factor)}


# ─────────────────────────────  Vistas  ──────────────────────────────
def _db_relativo(p):
    v = 10 * np.log10(np.maximum(p, 1e-12))
    return np.clip((v - np.median(v)) / 10.0, -3.0, 8.0)


def vista_espectro(x):
    _, p = _psd(x, min(BINS_ESPECTRO, len(x)))
    if len(p) < BINS_ESPECTRO:
        p = np.interp(np.linspace(0, len(p) - 1, BINS_ESPECTRO), np.arange(len(p)), p)
    return _db_relativo(p)[None]


def _espectro_lineas(y):
    grupo = int(np.ceil(len(y) / BINS_ESPECTRO))
    n = grupo * BINS_ESPECTRO
    s = np.abs(np.fft.fftshift(np.fft.fft(y * np.hanning(len(y)), n))) ** 2
    return _db_relativo(s.reshape(BINS_ESPECTRO, grupo).max(axis=1))


def vista_potencias(x):
    return np.stack([_espectro_lineas(x ** n) for n in POTENCIAS])


def vista_envolvente(x):
    a = np.abs(x)
    a = a / max(np.sqrt(np.mean(a ** 2)), 1e-30)
    h, _ = np.histogram(a, BINS_ENVOLVENTE, range=(0.0, 3.0))
    h = h / max(h.max(), 1)
    c = a - a.mean()
    varianza = max(float(np.dot(c, c)), 1e-30)
    autocorrelacion = np.array([np.dot(c[:-k], c[k:]) / varianza if k < len(c) else 0.0
                                for k in range(1, RETARDOS + 1)])
    return np.stack([h, autocorrelacion])


def vista_frecuencia(x):
    producto = x[1:] * np.conj(x[:-1])
    f = np.angle(producto) / (2 * np.pi)
    peso = np.abs(producto)
    suave = np.convolve(f, np.ones(SUAVIZADO) / SUAVIZADO, mode="same")
    salida = []
    for valores in (f, suave):
        h, _ = np.histogram(valores, BINS_FRECUENCIA, range=(-0.5, 0.5), weights=peso)
        salida.append(h / max(h.max(), 1e-30))
    return np.stack(salida)


def vistas(iq):
    """{nombre: arreglo (canales, largo)} de una señal I/Q cualquiera, más la información de banda."""
    x, banda = normalizar_banda(iq)
    if len(x) < LARGO_MIN:
        x = np.resize(x, LARGO_MIN)
    salida = {"espectro": vista_espectro(x), "potencias": vista_potencias(x),
              "envolvente": vista_envolvente(x), "frecuencia": vista_frecuencia(x)}
    return {k: v.astype(np.float32) for k, v in salida.items()}, banda
