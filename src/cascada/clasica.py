"""Receptor clásico de referencia: filtro adaptado, Oerder-Meyr y CMA fraccionado."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter1d
from scipy.signal import fftconvolve

from src.waveform import rrc_filter

SPS = 8
LARGO_FSE = 17


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


def ventanas_t2(x2, largo=LARGO_FSE):
    return np.lib.stride_tricks.sliding_window_view(x2, largo)[::2]


def peso_inicial(largo=LARGO_FSE):
    w = np.zeros(largo, dtype=np.complex128)
    w[largo // 2] = 1.0
    return w


def pesos_cma(x2, pasadas=3, mu=2e-3):
    """Pesos del ecualizador T/2; entrada de potencia unitaria."""
    w = peso_inicial()
    x2 = x2 / np.sqrt(np.mean(np.abs(x2) ** 2))
    ventanas = ventanas_t2(x2)
    if len(ventanas) <= 8:
        return w
    modulo = np.mean(np.abs(x2) ** 4) / np.mean(np.abs(x2) ** 2)
    for pasada in range(pasadas):
        paso = mu / (1 + pasada)
        for u in ventanas:
            y = np.dot(w, u)
            w = w - paso * y * (np.abs(y) ** 2 - modulo) * np.conj(u)
            if not np.isfinite(y):
                return peso_inicial()
    return w


def modos_constelacion(z, bins=96, limite=2.4, relativo=0.15):
    """Picos de densidad de la nube: un alfabeto estimado sin suponer modulación."""
    h, ey, ex = np.histogram2d(z.imag, z.real, bins=bins, range=[[-limite, limite], [-limite, limite]])
    h = gaussian_filter(h, 1.0)
    picos = (h == maximum_filter(h, size=5)) & (h > relativo * h.max())
    i, j = np.nonzero(picos)
    cy, cx = 0.5 * (ey[:-1] + ey[1:]), 0.5 * (ex[:-1] + ex[1:])
    return cx[j] + 1j * cy[i]


def refinar_ciego(x2, w, entropia, pasadas=4, ridge=1e-3):
    """Ecualización dirigida por los modos de la propia constelación."""
    x = ventanas_t2(x2)
    def salida(p):
        y = x @ p
        return y / np.sqrt(max(np.mean(np.abs(y) ** 2), 1e-12))
    mejor_w, y = w, salida(w)
    mejor_h = entropia(y)
    for _ in range(pasadas):
        modos = modos_constelacion(y)
        if len(modos) < 2:
            break
        d = np.abs(y[:, None] - modos[None, :])
        k = d.argmin(1)
        r = d[np.arange(len(y)), k]
        fiables = r < 2 * np.median(r)
        a = x[fiables]
        g = a.conj().T @ a
        g += ridge * np.trace(g).real / len(g) * np.eye(len(g))
        nuevo = np.linalg.solve(g, a.conj().T @ modos[k[fiables]])
        y_n = salida(nuevo)
        h = entropia(y_n)
        if not h < mejor_h:
            break
        mejor_h, mejor_w, y = h, nuevo, y_n
    return mejor_w


def simbolos_t2(onda):
    x = tomar_simbolos(onda, por_simbolo=2)
    return x / np.sqrt(max(np.mean(np.abs(x) ** 2), 1e-12))


def simbolos_sin_ecualizar(iq8):
    return tomar_simbolos(filtro_adaptado(iq8))


def simbolos_clasicos(iq8):
    x2 = simbolos_t2(filtro_adaptado(iq8))
    return ventanas_t2(x2) @ pesos_cma(x2)
