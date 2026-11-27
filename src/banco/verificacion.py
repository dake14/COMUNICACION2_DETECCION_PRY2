"""Medidas clásicas para comprobar que el banco genera lo que dice generar."""
import numpy as np
from scipy.signal import welch


def snr_medida(resultado):
    s, r = resultado["sin_ruido"], resultado["ruido"]
    return float(10 * np.log10(np.mean(np.abs(s) ** 2) / np.mean(np.abs(r) ** 2)))


def tasa_ciclica(iq, minima=1 / 32, maxima=0.45):
    """Línea más fuerte de |x|², en ciclos por muestra."""
    a = np.abs(np.asarray(iq, dtype=np.complex128)) ** 2
    espectro = np.abs(np.fft.rfft((a - a.mean()) * np.hanning(len(a)), 8 * len(a)))
    f = np.fft.rfftfreq(8 * len(a))
    validos = np.flatnonzero((f >= minima) & (f <= maxima))
    return float(f[validos[int(np.argmax(espectro[validos]))]])


def prominencia_ciclica(iq, frecuencia, derivada=False):
    """Altura de la línea en `frecuencia` frente a la mediana del espectro cercano."""
    x = np.asarray(iq, dtype=np.complex128)
    a = np.abs(np.diff(x, prepend=x[0]) if derivada else x) ** 2
    espectro = np.abs(np.fft.rfft((a - a.mean()) * np.hanning(len(a)), 8 * len(a)))
    k = int(round(frecuencia * 8 * len(a)))
    cerca = np.r_[espectro[max(0, k - 800):max(0, k - 32)], espectro[k + 32:k + 800]]
    return float(espectro[max(0, k - 16):k + 17].max() / max(np.median(cerca), 1e-30))


def ancho_ocupado(iq, relativo=0.01):
    """Ancho, en ciclos por muestra, donde la densidad supera una fracción de su nivel en banda."""
    f, p = welch(np.asarray(iq, dtype=np.complex128), nperseg=min(1024, len(iq)), return_onesided=False, detrend=False)
    orden = np.argsort(f)
    f, p = f[orden], p[orden]
    dentro = f[p > relativo * np.percentile(p, 95)]
    return float(dentro[-1] - dentro[0])
