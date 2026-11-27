"""Señales de entrenamiento del recuperador con canales variables e interferencias.

CONTRATO (unidad U2, cerrado):

Dependencias permitidas (lista cerrada): dataclasses.replace, numpy,
src.cascada.sintesis (SPS, condicion_aleatoria, transmitir), src.banco.canales_extra (TODOS_CANALES),
src.banco.canales (potencia).

- ``CANALES_ENTRENAMIENTO``: tupla con los nombres de ``TODOS_CANALES`` cuyo ``Canal.gnuradio`` es False,
  excluyendo "ideal" y "proyecto", en el orden del diccionario.

- ``condicion_limpia(rng) -> Condicion``: llama ``condicion_aleatoria(rng)`` y devuelve
  ``replace(c, taps=np.array([1 + 0j]), snr_db=300.0, bits=None, continua=0j, desbalance=0j)``.

- ``transmitir_variado(puntos, n_simbolos, rng, prob_canal=0.85, snr_rango=(-5.0, 30.0)) -> dict``:
  1. ``cond = condicion_limpia(rng)``; ``r = transmitir(puntos, n_simbolos, cond, rng)`` (sale a SPS = 8 muestras/símbolo).
  2. ``x = r["iq"].astype(np.complex128)``.
  3. Con probabilidad ``prob_canal`` (``rng.uniform() < prob_canal``) ``canal = str(rng.choice(CANALES_ENTRENAMIENTO))``;
     si no, ``canal = "ideal"``.
  4. ``medio = TODOS_CANALES[canal]``; ``y = medio.antes(x, float(SPS), rng)[0]`` si ``medio.antes`` no es None, si no ``y = x``.
  5. ``snr = float(rng.uniform(*snr_rango))``; ruido complejo gaussiano de potencia ``potencia(y) / 10**(snr/10)``
     (``sqrt(p/2) * (normal + j normal)`` con ``rng.normal``); ``z = y + ruido``.
  6. Si ``medio.despues`` no es None: ``z = medio.despues(z, float(SPS), rng)[0]``.
  7. ``z = z / sqrt(mean(|z|^2))``; si no es finito o la potencia es 0, ``z = x``.
  8. Devuelve ``{"iq": z complex64, "limpia": r["limpia"], "simbolos": r["simbolos"], "canal": canal, "snr_db": snr}``.
  Los arrays "iq" y "limpia" tienen el mismo largo.

Criterio de aceptación: con ``rng = np.random.default_rng(1)``, 200 llamadas con puntos QPSK y n_simbolos=132
devuelven iq y limpia del mismo largo (132*8 ± 8), finitos, iq con potencia 1 ± 1e-3, y aparecen al menos
10 canales distintos; la misma semilla reproduce la misma salida.
"""
from dataclasses import replace

import numpy as np

from src.banco.canales import potencia
from src.banco.canales_extra import TODOS_CANALES
from src.cascada.sintesis import SPS, condicion_aleatoria, transmitir

CANALES_ENTRENAMIENTO = tuple(
    nombre for nombre, canal in TODOS_CANALES.items()
    if not canal.gnuradio and nombre not in ("ideal", "proyecto")
)


def condicion_limpia(rng):
    c = condicion_aleatoria(rng)
    return replace(c, taps=np.array([1 + 0j]), snr_db=300.0, bits=None, continua=0j, desbalance=0j)


def transmitir_variado(puntos, n_simbolos, rng, prob_canal=0.85, snr_rango=(-5.0, 30.0)):
    cond = condicion_limpia(rng)
    r = transmitir(puntos, n_simbolos, cond, rng)
    x = r["iq"].astype(np.complex128)

    if rng.uniform() < prob_canal:
        canal = str(rng.choice(CANALES_ENTRENAMIENTO))
    else:
        canal = "ideal"

    medio = TODOS_CANALES[canal]
    y = medio.antes(x, float(SPS), rng)[0] if medio.antes is not None else x

    snr = float(rng.uniform(*snr_rango))
    p_ruido = potencia(y) / 10 ** (snr / 10)
    ruido = np.sqrt(p_ruido / 2) * (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y)))
    z = y + ruido

    if medio.despues is not None:
        z = medio.despues(z, float(SPS), rng)[0]

    p_z = potencia(z)
    if np.isfinite(p_z) and p_z > 0 and np.all(np.isfinite(z)):
        z = z / np.sqrt(p_z)
        if not np.all(np.isfinite(z)):
            z = x
    else:
        z = x

    return {
        "iq": z.astype(np.complex64),
        "limpia": r["limpia"],
        "simbolos": r["simbolos"],
        "canal": canal,
        "snr_db": snr,
    }
