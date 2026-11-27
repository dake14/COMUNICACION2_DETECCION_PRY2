"""Canales e imperfecciones. `antes` actúa antes del ruido y `despues` sobre la señal recibida."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline

from src.banco.isi import ajustar

CONFIG_PROYECTO = Path(__file__).resolve().parents[2] / "configs" / "dataset_e2.json"


@dataclass(frozen=True)
class Canal:
    antes: Optional[Callable] = None
    despues: Optional[Callable] = None
    gnuradio: bool = False


def potencia(x):
    return float(np.mean(np.abs(x) ** 2))


def perfil_rayos(sps, rng, retardo_simbolos, n_min, n_max):
    """Retardos en muestras y amplitudes con caída exponencial; el primer rayo llega en cero."""
    n = int(rng.integers(n_min, n_max + 1))
    retardos = np.r_[0.0, np.sort(rng.uniform(0.05, retardo_simbolos, n - 1))]
    tau = rng.uniform(0.3, 1.5) * retardo_simbolos
    magnitudes = np.exp(-retardos / (2 * tau))
    return retardos * sps, magnitudes / np.sqrt(np.sum(magnitudes ** 2))


def doppler_por_simbolo(rng, bajo=1e-4, alto=5e-3):
    return float(np.exp(rng.uniform(np.log(bajo), np.log(alto))))


def _retrasar(x, d):
    return np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * np.fft.fftfreq(len(x)) * d))


def _jakes(n, fd, rng, ondas=16):
    t = np.arange(n)
    theta = rng.uniform(0, 2 * np.pi, ondas)
    fase = rng.uniform(0, 2 * np.pi, ondas)
    return np.exp(1j * (2 * np.pi * fd * np.cos(theta)[:, None] * t + fase[:, None])).sum(0) / np.sqrt(ondas)


def _ganancia_rician(n, fd, k, rng):
    los = np.exp(2j * np.pi * rng.uniform())
    return np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * _jakes(n, fd, rng)


def _sumar_rayos(x, retardos, magnitudes, ganancias):
    y = np.zeros(len(x), dtype=np.complex128)
    for d, m, g in zip(retardos, ajustar(np.asarray(magnitudes, float)), ganancias):
        y += m * g * _retrasar(x, d)
    return y


def _estatico(retardo_simbolos, n_min, n_max):
    def canal(x, sps, rng):
        d, m = perfil_rayos(sps, rng, retardo_simbolos, n_min, n_max)
        g = np.exp(2j * np.pi * rng.uniform(size=len(d)))
        return _sumar_rayos(x, d, m, g), {"retardos": d.tolist(), "magnitudes": m.tolist()}
    return canal


def _tdl(rician):
    def canal(x, sps, rng):
        d, m = perfil_rayos(sps, rng, 3.0, 3, 6)
        fd = doppler_por_simbolo(rng)
        k_db = float(rng.uniform(3, 15)) if rician else None
        g = [_jakes(len(x), fd / sps, rng) for _ in d]
        if rician:
            g[0] = _ganancia_rician(len(x), fd / sps, 10 ** (k_db / 10), rng)
        return _sumar_rayos(x, d, m, g), {"retardos": d.tolist(), "magnitudes": m.tolist(),
                                          "doppler_por_simbolo": fd, "k_db": k_db}
    return canal


def ideal(x, sps, rng):
    return x, {}


def proyecto(x, sps, rng):
    datos = json.loads(CONFIG_PROYECTO.read_text(encoding="utf-8"))
    canales = datos["id_channels"][1:] + datos["shift_channels"]
    i = int(rng.integers(len(canales)))
    taps = ajustar(np.array([complex(a, b) for a, b in canales[i]]))
    return np.convolve(x, taps)[:len(x)], {"indice": i}


def plano_lento(x, sps, rng):
    fd = doppler_por_simbolo(rng, 1e-4, 1e-3)
    k_db = float(rng.uniform(0, 10))
    return x * _ganancia_rician(len(x), fd / sps, 10 ** (k_db / 10), rng), {"doppler_por_simbolo": fd, "k_db": k_db}


def ruido_fase(x, sps, rng):
    grados = float(rng.uniform(0.2, 2.5))
    pasos = np.deg2rad(grados) / np.sqrt(sps) * rng.normal(size=len(x))
    return x * np.exp(1j * np.cumsum(pasos)), {"grados_por_simbolo": grados}


def deriva_portadora(x, sps, rng):
    total = float(rng.uniform(1e-4, 2e-3) * rng.choice([-1, 1]))
    n = np.arange(len(x), dtype=float)
    fase = np.pi * total / sps * n ** 2 / len(x)
    paseo = np.cumsum(rng.normal(0, 2e-3 / np.sqrt(sps), len(x)))
    return x * np.exp(1j * (fase + paseo)), {"deriva_ciclos_por_simbolo": total}


def deriva_reloj(x, sps, rng):
    ppm = float(rng.uniform(-200, 200))
    variacion = np.cumsum(rng.normal(0, 2e-7, len(x)))
    t = np.arange(len(x), dtype=float) * (1 + ppm * 1e-6) + np.cumsum(variacion)
    t = np.clip(t, 0, len(x) - 1)
    return CubicSpline(np.arange(len(x), dtype=float), x)(t), {"ppm": ppm}


def iq_dc(x, sps, rng):
    g = 10 ** (float(rng.uniform(0.1, 1.5)) / 20)
    fase = np.deg2rad(rng.uniform(0.5, 8))
    e = (g - 1) / (g + 1)
    a = np.cos(fase / 2) + 1j * e * np.sin(fase / 2)
    b = e * np.cos(fase / 2) - 1j * np.sin(fase / 2)
    dc = rng.uniform(0, 0.05) * np.sqrt(potencia(x)) * np.exp(2j * np.pi * rng.uniform())
    return a * x + b * np.conj(x) + dc, {"ganancia": float(g), "fase_rad": float(fase), "dc": abs(complex(dc))}


def amplificador(x, sps, rng):
    retroceso = float(rng.uniform(1, 8))
    p = float(rng.uniform(1, 3))
    r = np.abs(x) / (np.sqrt(potencia(x)) * 10 ** (retroceso / 20))
    ganancia = (1 + r ** (2 * p)) ** (-1 / (2 * p))
    alfa = float(rng.uniform(0, 0.6))
    return x * ganancia * np.exp(1j * alfa * r ** 2 / (1 + r ** 2)), {"retroceso_db": retroceso, "p": p, "alfa": alfa}


def interferencia(x, sps, rng):
    sir = float(rng.uniform(10, 25))
    n = np.arange(len(x))
    desvio = float(rng.uniform(0.6, 1.2) * rng.choice([-1, 1]) / sps)
    tipo = str(rng.choice(["tono", "banda"]))
    if tipo == "tono":
        i = np.exp(2j * np.pi * desvio * n)
    else:
        f = np.fft.fftfreq(len(x))
        ruido = rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))
        i = np.fft.ifft(np.fft.fft(ruido) * (np.abs(f) < 0.25 / sps)) * np.exp(2j * np.pi * desvio * n)
    i *= np.sqrt(potencia(x) / max(potencia(i), 1e-20) / 10 ** (sir / 10))
    return x + i, {"sir_db": sir, "tipo": tipo, "desvio": desvio}


def impulsivo(z, sps, rng):
    p = float(rng.uniform(1e-3, 1e-2))
    escala = float(rng.uniform(3, 10))
    golpes = rng.uniform(size=len(z)) < p
    ruido = rng.normal(size=len(z)) + 1j * rng.normal(size=len(z))
    return z + golpes * ruido * escala * np.sqrt(potencia(z) / 2), {"probabilidad": p, "escala": escala}


def cuantizacion(z, sps, rng):
    bits = int(rng.integers(4, 11))
    limite = float(rng.uniform(1.5, 4)) * np.sqrt(potencia(z) / 2)
    paso = 2 * limite / 2 ** bits
    cuant = lambda v: np.clip(np.round(v / paso) * paso, -limite, limite)
    return cuant(z.real) + 1j * cuant(z.imag), {"bits": bits, "limite": limite}


def _encadenar(*etapas):
    def canal(x, sps, rng):
        parametros = {}
        for etapa in etapas:
            x, p = etapa(x, sps, rng)
            parametros[etapa.__name__] = p
        return x, parametros
    return canal


def _gnuradio(nombre):
    def canal(x, sps, rng):
        from src.banco import gr_bloques
        return getattr(gr_bloques, nombre)(x, sps, rng)
    return canal


CANALES = {
    "ideal": Canal(ideal),
    "proyecto": Canal(proyecto),
    "mp_corto": Canal(_estatico(1.0, 2, 3)),
    "mp_largo": Canal(_estatico(4.0, 3, 6)),
    "tdl_rayleigh": Canal(_tdl(False)),
    "tdl_rician": Canal(_tdl(True)),
    "plano_lento": Canal(plano_lento),
    "ruido_fase": Canal(ruido_fase),
    "deriva_portadora": Canal(deriva_portadora),
    "deriva_reloj": Canal(deriva_reloj),
    "iq_dc": Canal(iq_dc),
    "amplificador": Canal(amplificador),
    "interferencia": Canal(interferencia),
    "impulsivo": Canal(despues=impulsivo),
    "cuantizacion": Canal(despues=cuantizacion),
    "receptor_barato": Canal(_encadenar(ruido_fase, deriva_portadora, iq_dc), cuantizacion),
    "gr_multitrayecto": Canal(_gnuradio("canal_multitrayecto"), gnuradio=True),
    "gr_selectivo": Canal(_gnuradio("canal_selectivo"), gnuradio=True),
    "gr_dinamico": Canal(_gnuradio("canal_dinamico"), gnuradio=True),
    "gr_imperfecciones": Canal(_gnuradio("canal_imperfecciones"), gnuradio=True),
}
