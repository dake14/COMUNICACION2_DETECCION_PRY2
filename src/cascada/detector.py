"""Detección y SNR ciegas a partir de la forma del espectro."""
from functools import lru_cache

import numpy as np
from scipy.signal import welch


def _psd(x):
    nperseg = int(min(len(x), 256))
    _, p = welch(x, nperseg=nperseg, return_onesided=False, detrend=False)
    return p


def estadistico_espectral(x):
    """Cociente entre la zona más fuerte del espectro y su mediana."""
    p = _psd(np.asarray(x, dtype=np.complex128))
    k = max(1, len(p) // 8)
    return float(np.mean(np.sort(p)[-k:]) / max(np.median(p), 1e-30))


@lru_cache(maxsize=64)
def umbral_ruido_blanco(n, pfa=0.01, ensayos=500):
    rng = np.random.default_rng(20261118 + n)
    valores = [estadistico_espectral(rng.normal(size=n) + 1j * rng.normal(size=n)) for _ in range(ensayos)]
    return float(np.quantile(valores, 1 - pfa))


def detectar_ciego(iq, pfa=0.01):
    x = np.asarray(iq, dtype=np.complex128)
    x = x - x.mean()
    estadistico = estadistico_espectral(x)
    umbral = umbral_ruido_blanco(len(x), pfa)
    return {"detectada": bool(estadistico > umbral), "estadistico": estadistico, "umbral": umbral}


def snr_ciega_db(iq):
    """SNR por muestra usando el piso espectral como estimación del ruido."""
    x = np.asarray(iq, dtype=np.complex128)
    x = x - x.mean()
    p = _psd(x)
    piso = np.percentile(p, 15) * 1.25
    total = np.mean(p)
    exceso = total - piso
    if exceso <= 0 or piso <= 0:
        return None
    return float(10 * np.log10(exceso / piso))
