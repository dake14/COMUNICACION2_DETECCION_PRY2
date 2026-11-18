"""Etapa 1: estima tasa de símbolo y CFO grueso, y lleva la señal a 8 muestras/símbolo."""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks, resample_poly, welch

SPS_OBJETIVO = 8
SPS_MIN = 2.2
SPS_MAX = 32.0


@dataclass
class Hipotesis:
    iq: np.ndarray
    sps_origen: float
    puntaje: float
    cfo_grueso: float


def _espectro_exceso(x):
    nperseg = int(min(len(x), max(256, min(2048, len(x) // 4))))
    f, p = welch(x, nperseg=nperseg, return_onesided=False, detrend=False)
    orden = np.argsort(f)
    f, p = f[orden], p[orden]
    p = uniform_filter1d(p, max(3, nperseg // 64), mode="wrap")
    piso = np.percentile(p, 15)
    return f, np.maximum(p - piso, 0.0)


def _cfo_y_ancho(x):
    f, exceso = _espectro_exceso(x)
    total = exceso.sum()
    if total <= 0:
        return 0.0, 0.5
    acumulado = np.cumsum(exceso) / total
    bajo, alto = np.interp([0.02, 0.98], acumulado, f)
    dentro = (f >= bajo) & (f <= alto)
    mitad = exceso[dentro] > 0.25 * exceso[dentro].max()
    borde = f[dentro][mitad]
    return float(0.5 * (borde[0] + borde[-1])), float(max(alto - bajo, 1e-3))


def _lineas_ciclicas(x):
    n = len(x)
    relleno = 8 * n
    total = np.zeros(relleno // 2 + 1)
    for v in (np.abs(x) ** 2, np.abs(np.diff(x, prepend=x[0])) ** 2):
        v = (v - v.mean()) * np.hanning(n)
        s = np.abs(np.fft.rfft(v, relleno))
        ancho = max(33, relleno // 64) | 1
        piso = median_filter(s[:: 4], size=max(5, ancho // 4 | 1), mode="nearest")
        piso = np.interp(np.arange(len(s)), np.arange(len(piso)) * 4, piso)
        total += s / np.maximum(piso, 1e-12)
    return np.fft.rfftfreq(relleno), total / 2


def _candidatos(x, ancho, maximo):
    f, s = _lineas_ciclicas(x)
    validos = (f >= 1 / SPS_MAX) & (f <= 1 / SPS_MIN)
    picos, _ = find_peaks(np.where(validos, s, 0.0), distance=max(4, len(f) // 800))
    salida = []
    for k in picos:
        a, b, c = s[k - 1], s[k], s[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        tasa = f[k] + d * (f[1] - f[0])
        puntaje = float(b)
        if not 0.4 * ancho <= tasa <= 1.25 * ancho:
            puntaje *= 0.3
        salida.append((puntaje, tasa))
    salida.sort(reverse=True)
    return salida[:maximo]


def _ajustar_sps(sps):
    entero = round(sps)
    return float(entero) if entero >= 2 and abs(sps - entero) / sps < 0.004 else float(sps)


def a_ocho_sps(x, sps):
    if sps == SPS_OBJETIVO:
        return x
    r = Fraction(SPS_OBJETIVO / sps).limit_denominator(512)
    return resample_poly(x, r.numerator, r.denominator)


def preparar(iq):
    x = np.asarray(iq, dtype=np.complex128)
    x = x - x.mean()
    potencia = np.mean(np.abs(x) ** 2)
    if not np.isfinite(potencia) or potencia <= 0:
        raise ValueError("La señal no tiene potencia útil.")
    return x / np.sqrt(potencia)


def normalizar(iq, sps=None, max_hipotesis=3):
    x = preparar(iq)
    cfo, ancho = _cfo_y_ancho(x)
    x = x * np.exp(-2j * np.pi * cfo * np.arange(len(x)))
    if sps is not None:
        lista = [(np.inf, 1.0 / float(sps))]
    else:
        lista = _candidatos(x, ancho, max_hipotesis)
        if not lista:
            lista = [(0.0, min(1 / SPS_MIN, max(1 / SPS_MAX, ancho / 1.35)))]
    hipotesis = []
    for puntaje, tasa in lista:
        s = _ajustar_sps(1.0 / tasa)
        y = a_ocho_sps(x, s)
        y = y / np.sqrt(np.mean(np.abs(y) ** 2))
        hipotesis.append(Hipotesis(y.astype(np.complex64), s, float(puntaje), cfo))
    return hipotesis
