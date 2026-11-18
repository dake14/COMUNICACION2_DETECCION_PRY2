"""Transmisor y canal procedural con referencia limpia para el recuperador."""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from src.waveform import rrc_filter

SPS = 8
BETA_CANONICO = 0.35
SPAN_CANONICO = 16


def coseno_alzado(beta=BETA_CANONICO, sps=SPS, span=SPAN_CANONICO):
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    denominador = 1 - (2 * beta * t) ** 2
    singular = np.isclose(denominador, 0.0)
    seguro = np.where(singular, 1.0, denominador)
    h = np.sinc(t) * np.cos(np.pi * beta * t) / seguro
    return np.where(singular, np.pi / 4 * np.sinc(1 / (2 * beta)), h)


def pulso(tipo, beta, span, sps=SPS):
    if tipo == "rrc":
        h = rrc_filter(beta=beta, sps=sps, span=span)
    elif tipo == "rc":
        h = coseno_alzado(beta, sps, span)
    elif tipo == "gauss":
        t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
        h = np.exp(-2 * (np.pi * (0.25 + 0.5 * beta) * t) ** 2 / np.log(2))
    elif tipo == "semiseno":
        h = np.sin(np.pi * (np.arange(2 * sps) + 0.5) / (2 * sps))
    elif tipo == "kaiser":
        t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
        h = np.sinc(t) * np.kaiser(len(t), 4 + 6 * beta)
    else:
        raise ValueError(f"Pulso desconocido: {tipo}")
    return h / np.sqrt(np.sum(h ** 2))


@dataclass
class Condicion:
    snr_db: float = 30.0
    pulso: str = "rrc"
    beta: float = 0.35
    span: int = 10
    taps: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j]))
    cfo: float = 0.0
    fase: float = 0.0
    tau: float = 0.0
    error_tasa: float = 0.0
    desbalance: complex = 0j
    continua: complex = 0j
    bits: int | None = None
    forma: float = 0.0


def canal_aleatorio(rng, retardo_max=24, trayectos_max=6):
    n = int(rng.integers(2, trayectos_max + 1))
    retardos = np.concatenate(([0], np.sort(rng.integers(1, retardo_max + 1, n - 1))))
    decaimiento = np.exp(-retardos / rng.uniform(4, 24))
    gan = (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(decaimiento / 2)
    taps = np.zeros(retardo_max + 1, dtype=np.complex128)
    np.add.at(taps, retardos, gan)
    return taps[: retardos[-1] + 1]


def condicion_aleatoria(rng, snr_rango=(-5.0, 30.0), cfo_max=0.003, prob_canal=0.75,
                        retardo_max=24):
    tipo = rng.choice(["rrc", "rc", "gauss", "semiseno", "kaiser"],
                      p=[0.55, 0.15, 0.1, 0.1, 0.1])
    taps = canal_aleatorio(rng, retardo_max) if rng.uniform() < prob_canal else np.array([1 + 0j])
    bits = int(rng.integers(8, 13)) if rng.uniform() < 0.3 else None
    return Condicion(
        snr_db=float(rng.uniform(*snr_rango)), pulso=str(tipo), beta=float(rng.uniform(0.1, 1.0)),
        span=int(rng.choice([6, 8, 10, 12, 16])), taps=taps,
        cfo=float(rng.uniform(-cfo_max, cfo_max)), fase=float(rng.uniform(-np.pi, np.pi)),
        tau=float(rng.uniform(0, SPS)), error_tasa=float(rng.uniform(-1e-3, 1e-3)),
        desbalance=complex(rng.uniform(0, 0.03) * np.exp(2j * np.pi * rng.uniform())),
        continua=complex(rng.uniform(0, 0.02) * np.exp(2j * np.pi * rng.uniform())),
        bits=bits, forma=float(rng.uniform(0, 1.5)) if rng.uniform() < 0.2 else 0.0,
    )


def _cuantizar(x, bits):
    escala = 3.0 * np.sqrt(np.mean(np.abs(x) ** 2) / 2)
    paso = 2 * escala / 2 ** bits
    cuant = lambda v: np.clip(np.round(v / paso) * paso, -escala, escala)
    return cuant(x.real) + 1j * cuant(x.imag)


def transmitir(puntos, n_simbolos, cond, rng, sps_salida=SPS):
    """Devuelve iq con potencia 1, y `limpia` con pulso canónico y misma sincronía."""
    puntos = np.asarray(puntos)
    p = np.exp(-cond.forma * np.abs(puntos) ** 2)
    guarda = 40
    total = n_simbolos + 2 * guarda
    simbolos = puntos[rng.choice(len(puntos), total, p=p / p.sum())]
    tren = np.zeros(total * SPS, dtype=np.complex128)
    tren[::SPS] = simbolos

    h = pulso(cond.pulso, cond.beta, cond.span)
    rc = coseno_alzado()
    x = fftconvolve(tren, h)[(len(h) - 1) // 2:][: len(tren)]
    limpia = fftconvolve(tren, rc)[(len(rc) - 1) // 2:][: len(tren)]

    taps = np.asarray(cond.taps, dtype=np.complex128)
    principal = int(np.argmax(np.abs(taps)))
    y = fftconvolve(x, taps)[: len(tren)]
    limpia = np.roll(limpia, principal) * np.exp(1j * np.angle(taps[principal]))

    paso = (SPS / sps_salida) * (1 + cond.error_tasa)
    n_salida = int(n_simbolos * SPS / paso)
    inicio = guarda * SPS + cond.tau
    t = inicio + paso * np.arange(n_salida)
    base = np.arange(len(tren), dtype=float)
    y = CubicSpline(base, y)(t)
    limpia = CubicSpline(base, limpia)(t)

    giro = np.exp(1j * (cond.fase + 2 * np.pi * cond.cfo * np.arange(n_salida)))
    y, limpia = y * giro, limpia * giro
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))
    y = y + cond.desbalance * np.conj(y)

    potencia_ruido = np.mean(np.abs(y) ** 2) / 10 ** (cond.snr_db / 10)
    ruido = np.sqrt(potencia_ruido / 2) * (rng.normal(size=n_salida) + 1j * rng.normal(size=n_salida))
    iq = y + ruido + cond.continua
    if cond.bits is not None:
        iq = _cuantizar(iq, cond.bits)
    iq = iq / np.sqrt(np.mean(np.abs(iq) ** 2))

    indices = guarda + np.arange(n_simbolos)
    return {"iq": iq.astype(np.complex64), "limpia": limpia.astype(np.complex64),
            "simbolos": simbolos[indices].astype(np.complex64)}
