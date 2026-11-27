"""Transmisores de fase continua (FSK, MSK, GMSK): uno por generador del banco."""
# ─────────────────────────────  Imports  ─────────────────────────────
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import bessel, butter, sosfilt

from src.banco.generadores import SOBRE_ANALOGICO, SOBRE_FRECUENCIA, _a_sps

# ─────────────────────────────  Config  ──────────────────────────────
SOBRE_PROCEDURAL = 16
SPAN_GAUSS = 2


# ───────────────────────────────  Pulso  ─────────────────────────────
def pulso_frecuencia(sobre, bt=None):
    """Pulso de frecuencia de un símbolo (rectangular o rectangular ⊗ gaussiano) con área `sobre`."""
    rect = np.ones(sobre)
    if bt is None:
        return rect
    t = np.arange(-SPAN_GAUSS * sobre, SPAN_GAUSS * sobre + 1) / sobre
    gauss = np.exp(-2 * np.pi ** 2 * bt ** 2 * t ** 2 / np.log(2))
    g = np.convolve(gauss, rect)
    return g * sobre / g.sum()


def niveles(orden, n, rng):
    return (2 * rng.integers(0, orden, n) - (orden - 1)).astype(float)


def frecuencia_instantanea(a, sobre, bt=None):
    """Tren de frecuencia: cada nivel ocupa su símbolo, centrado aunque el pulso sea largo."""
    tren = np.zeros(len(a) * sobre)
    tren[::sobre] = a
    g = pulso_frecuencia(sobre, bt)
    atraso = (len(g) - sobre) // 2
    return np.convolve(tren, g)[atraso:atraso + len(tren)]


def _modular(f, h, sobre):
    return np.exp(1j * np.pi * h * np.cumsum(f) / sobre)


# ──────────────────────────  Transmisores  ───────────────────────────
def repo(a, h, bt, cfg):
    sobre = int(round(cfg.sps))
    return _modular(frecuencia_instantanea(a, sobre, bt), h, sobre)


def procedural(a, h, bt, cfg):
    f = frecuencia_instantanea(a, SOBRE_PROCEDURAL, bt)
    fase = np.pi * h * np.cumsum(f) / SOBRE_PROCEDURAL
    t = np.arange(0.0, len(fase) - 1, SOBRE_PROCEDURAL / cfg.sps)
    return np.exp(1j * CubicSpline(np.arange(len(fase), dtype=float), fase)(t))


def frecuencia(a, h, bt, cfg):
    tren = np.zeros(len(a) * SOBRE_FRECUENCIA)
    tren[::SOBRE_FRECUENCIA] = a
    g = pulso_frecuencia(SOBRE_FRECUENCIA, bt)
    atraso = (len(g) - SOBRE_FRECUENCIA) // 2
    n = len(tren) + len(g)
    f = np.fft.irfft(np.fft.rfft(tren, n) * np.fft.rfft(g, n), n)[atraso:atraso + len(tren)]
    return _a_sps(_modular(f, h, SOBRE_FRECUENCIA), SOBRE_FRECUENCIA, cfg.sps)


def analogico(a, h, bt, cfg):
    """NRZ, filtro de premodulación tipo analógico y VCO."""
    f = np.repeat(a, SOBRE_ANALOGICO)
    ancho = bt if bt is not None else 0.6 + 0.8 * cfg.beta
    disenar = bessel if bt is not None or cfg.pulso == "nrz_bessel" else butter
    f = sosfilt(disenar(max(2, cfg.span // 2), 2 * ancho / SOBRE_ANALOGICO, output="sos"), f)
    return _a_sps(_modular(f, h, SOBRE_ANALOGICO), SOBRE_ANALOGICO, cfg.sps)


def _gnuradio(nombre):
    def crear(a, h, bt, cfg):
        from src.banco import gr_bloques
        return getattr(gr_bloques, nombre)(a, h, bt, cfg)
    return crear


TRANSMISORES = {
    "repo": repo,
    "procedural": procedural,
    "frecuencia": frecuencia,
    "analogico": analogico,
    "gr_polifase": _gnuradio("tx_cpm_polifase"),
    "gr_fir_mmse": _gnuradio("tx_cpm_fir_mmse"),
}
