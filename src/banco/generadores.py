"""Transmisores: convierten símbolos en banda base con implementaciones distintas."""
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import bessel, butter, fftconvolve, resample_poly, sosfilt

from src.cascada.sintesis import SPS as SPS_PROCEDURAL, pulso
from src.waveform import pulse_shape

SOBRE_FRECUENCIA = 48
SOBRE_ANALOGICO = 16


@dataclass(frozen=True)
class Generador:
    crear: Callable
    pulsos: tuple
    fraccionario: bool
    gnuradio: bool = False


def _a_sps(x, sobre, sps):
    r = Fraction(sps / sobre).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator)


def repo(simbolos, cfg, rng):
    x, h = pulse_shape(simbolos, sps=int(cfg.sps), beta=cfg.beta, span=cfg.span)
    return x[(len(h) - 1) // 2:]


def procedural(simbolos, cfg, rng):
    tren = np.zeros(len(simbolos) * SPS_PROCEDURAL, dtype=np.complex128)
    tren[::SPS_PROCEDURAL] = simbolos
    h = pulso(cfg.pulso, cfg.beta, cfg.span)
    x = fftconvolve(tren, h)[(len(h) - 1) // 2:][:len(tren)]
    t = np.arange(0.0, len(x) - 1, SPS_PROCEDURAL / cfg.sps)
    return CubicSpline(np.arange(len(x), dtype=float), x)(t)


def _respuesta(u, beta, tipo):
    if tipo == "gauss":
        return np.exp(-(u / (0.45 + beta)) ** 2)
    if tipo == "coseno_truncado":
        return np.where(u <= 0.5 + beta / 2, 1.0, 0.0) * np.cos(np.pi * np.minimum(u, 1) / (2 + 2 * beta))
    h = np.ones_like(u)
    zona = (u > (1 - beta) / 2) & (u <= (1 + beta) / 2)
    h[zona] = 0.5 * (1 + np.cos(np.pi / beta * (u[zona] - (1 - beta) / 2)))
    h[u > (1 + beta) / 2] = 0
    return np.sqrt(h)


def frecuencia(simbolos, cfg, rng):
    tren = np.zeros(len(simbolos) * SOBRE_FRECUENCIA, dtype=np.complex128)
    tren[::SOBRE_FRECUENCIA] = simbolos
    u = np.abs(np.fft.fftfreq(len(tren))) * SOBRE_FRECUENCIA
    x = np.fft.ifft(np.fft.fft(tren) * _respuesta(u, cfg.beta, cfg.pulso))
    return _a_sps(x, SOBRE_FRECUENCIA, cfg.sps)


def analogico(simbolos, cfg, rng):
    """Retención de orden cero seguida de un filtro de reconstrucción tipo analógico."""
    x = np.repeat(simbolos, SOBRE_ANALOGICO)
    corte = 2 * (0.6 + 0.8 * cfg.beta) / SOBRE_ANALOGICO
    orden = max(2, cfg.span // 2)
    disenar = butter if cfg.pulso == "nrz_butter" else bessel
    return _a_sps(sosfilt(disenar(orden, corte, output="sos"), x), SOBRE_ANALOGICO, cfg.sps)


def _gnuradio(nombre):
    def crear(simbolos, cfg, rng):
        from src.banco import gr_bloques
        return getattr(gr_bloques, nombre)(simbolos, cfg)
    return crear


GENERADORES = {
    "repo": Generador(repo, ("rrc",), False),
    "procedural": Generador(procedural, ("rrc", "rc", "gauss", "semiseno", "kaiser"), True),
    "frecuencia": Generador(frecuencia, ("rrc", "gauss", "coseno_truncado"), True),
    "analogico": Generador(analogico, ("nrz_butter", "nrz_bessel"), True),
    "gr_polifase": Generador(_gnuradio("tx_polifase"), ("rrc", "gauss"), True, True),
    "gr_fir_mmse": Generador(_gnuradio("tx_fir_mmse"), ("rrc", "gauss"), True, True),
}
