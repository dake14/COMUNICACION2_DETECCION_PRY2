"""Receptor clásico de referencia: filtro adaptado, Oerder-Meyr y CMA fraccionado."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve

from src.waveform import rrc_filter

SPS = 8


def fase_muestreo(onda, sps=SPS, ventana=32):
    """Instante de símbolo (en muestras) a lo largo de la señal."""
    n = np.arange(len(onda))
    c = (np.abs(onda) ** 2) * np.exp(-2j * np.pi * n / sps)
    largo = min(len(onda), ventana * sps)
    suave = uniform_filter1d(c.real, largo, mode="nearest") + 1j * uniform_filter1d(c.imag, largo, mode="nearest")
    return -np.unwrap(np.angle(suave)) * sps / (2 * np.pi)


def interpolar(onda, posiciones):
    return CubicSpline(np.arange(len(onda), dtype=float), onda)(posiciones)


def tomar_simbolos(onda, sps=SPS, por_simbolo=1):
    """Muestrea en los instantes de símbolo seguidos con Oerder-Meyr."""
    onda = np.asarray(onda, dtype=np.complex128)
    tau = fase_muestreo(onda, sps)
    base = tau[0] % sps
    posiciones = np.arange(base, len(onda) - sps, sps / por_simbolo)
    desvio = np.interp(posiciones, np.arange(len(tau)), tau - tau[0])
    posiciones = np.clip(posiciones + desvio, 0, len(onda) - 1)
    return interpolar(onda, posiciones)


def filtro_adaptado(iq8, beta=0.35, span=10):
    h = rrc_filter(beta=beta, sps=SPS, span=span)
    return fftconvolve(np.asarray(iq8, dtype=np.complex128), h, mode="same")


def cma_fraccionado(x2, medio=6, pasadas=3, mu=2e-3):
    """Ecualizador T/2 con salida a un símbolo; entrada de potencia unitaria."""
    largo = 2 * medio + 1
    w = np.zeros(largo, dtype=np.complex128)
    w[medio] = 1.0
    x2 = x2 / np.sqrt(np.mean(np.abs(x2) ** 2))
    n = (len(x2) - largo) // 2
    if n <= 8:
        return x2[::2]
    ventanas = np.lib.stride_tricks.sliding_window_view(x2, largo)[::2][:n]
    modulo = np.mean(np.abs(x2) ** 4) / np.mean(np.abs(x2) ** 2)
    for pasada in range(pasadas):
        paso = mu / (1 + pasada)
        for u in ventanas:
            y = np.dot(w, u)
            w = w - paso * y * (np.abs(y) ** 2 - modulo) * np.conj(u)
            if not np.isfinite(y):
                w = np.zeros(largo, dtype=np.complex128)
                w[medio] = 1.0
                return x2[::2]
    return ventanas @ w


def simbolos_sin_ecualizar(iq8):
    return tomar_simbolos(filtro_adaptado(iq8))


def simbolos_clasicos(iq8):
    x2 = tomar_simbolos(filtro_adaptado(iq8), por_simbolo=2)
    return cma_fraccionado(x2)
