"""Generador de prueba escrito aparte: pulsos en frecuencia y desvanecimiento variable."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.cascada import geometria as g

DESTINO = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cascada_prueba/independiente")
SEMILLA = 2026111902
SNRS = (5, 10, 15, 20, 30)
POR_CASO = 8
SOBRE = 48


def respuesta_pulso(f, rs, beta, tipo):
    u = np.abs(f) / rs
    if tipo == "coseno_alzado":
        h = np.ones_like(u)
        zona = (u > (1 - beta) / 2) & (u <= (1 + beta) / 2)
        h[zona] = 0.5 * (1 + np.cos(np.pi / beta * (u[zona] - (1 - beta) / 2)))
        h[u > (1 + beta) / 2] = 0
        return np.sqrt(h)
    if tipo == "gaussiano":
        return np.exp(-(u / (0.45 + beta)) ** 2)
    return np.where(u <= 0.5 + beta / 2, 1.0, 0.0) * np.cos(np.pi * np.minimum(u, 1) / (2 + 2 * beta))


def formar(simbolos, beta, tipo):
    tren = np.zeros(len(simbolos) * SOBRE, dtype=np.complex128)
    tren[::SOBRE] = simbolos
    f = np.fft.fftfreq(len(tren))
    return np.fft.ifft(np.fft.fft(tren) * respuesta_pulso(f, 1.0 / SOBRE, beta, tipo))


def desvanecimiento(n, doppler, rng, ondas=16):
    t = np.arange(n)
    theta = rng.uniform(0, 2 * np.pi, ondas)
    fase = rng.uniform(0, 2 * np.pi, ondas)
    suma = np.exp(1j * (2 * np.pi * doppler * np.cos(theta)[:, None] * t + fase[:, None])).sum(0)
    return suma / np.sqrt(ondas)


def canal_variable(x, sps, rng):
    if rng.uniform() < 0.3:
        return x
    n_rayos = int(rng.integers(2, 5))
    retardos = np.sort(rng.uniform(0, 2.5 * sps, n_rayos))
    retardos[0] = 0.0
    salida = np.zeros(len(x), dtype=np.complex128)
    for i, d in enumerate(retardos):
        ganancia = 10 ** (-(d / sps) * rng.uniform(2, 6) / 20)
        f = np.fft.fftfreq(len(x))
        rayo = np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * f * d))
        variacion = desvanecimiento(len(x), rng.uniform(0, 2e-5), rng) if i else 1.0
        salida += ganancia * rng.choice([1, -1]) * np.exp(2j * np.pi * rng.uniform()) * variacion * rayo
    return salida


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    grupos = {**{k: (v, k) for k, v in g.catalogo().items()},
              **{k: (v, "DESCONOCIDA") for k, v in g.desconocidas_prueba().items()}}
    manifiesto = []
    for nombre, (puntos, etiqueta) in grupos.items():
        for snr in SNRS:
            for _ in range(POR_CASO):
                sps = int(rng.choice([3, 4, 6, 8, 12]))
                n_sim = int(rng.choice([512, 1024]))
                beta = float(rng.choice([0.15, 0.3, 0.45, 0.7]))
                tipo = str(rng.choice(["coseno_alzado", "gaussiano", "coseno_truncado"], p=[0.7, 0.15, 0.15]))
                simbolos = puntos[rng.integers(0, len(puntos), n_sim + 64)]
                x = formar(simbolos, beta, tipo)
                x = np.roll(x, int(rng.integers(0, SOBRE)))
                x = resample_poly(x, sps, SOBRE)
                x = canal_variable(x, sps, rng)
                x = x[32 * sps: 32 * sps + n_sim * sps]
                n = np.arange(len(x))
                x = x * np.exp(1j * (2 * np.pi * rng.uniform(-0.012, 0.012) * n + rng.uniform(-np.pi, np.pi)))
                potencia = np.mean(np.abs(x) ** 2)
                ruido = np.sqrt(potencia / 10 ** (snr / 10) / 2)
                y = x + ruido * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
                if rng.uniform() < 0.5:
                    pico = np.percentile(np.abs(np.r_[y.real, y.imag]), 99.9)
                    y = np.round(y.real / pico * 2047) + 1j * np.round(y.imag / pico * 2047)
                archivo = f"i{len(manifiesto):04d}.c64"
                y.astype(np.complex64).tofile(DESTINO / archivo)
                manifiesto.append({"archivo": archivo, "real": nombre, "etiqueta": etiqueta, "snr": snr,
                                   "sps": sps, "beta": beta, "pulso": tipo, "n_simbolos": n_sim})
    (DESTINO / "manifiesto.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    print(len(manifiesto), "archivos")


if __name__ == "__main__":
    main()
