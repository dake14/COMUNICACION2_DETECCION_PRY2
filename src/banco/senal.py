"""Une símbolos, transmisor, canal y ruido en una señal I/Q."""
import importlib.util

import numpy as np

from src.banco.canales import potencia
from src.banco.canales_extra import TODOS_CANALES
from src.banco.generadores import GENERADORES
from src.banco.modulaciones import Modulacion

GUARDA = 40


def gnuradio_disponible():
    return importlib.util.find_spec("gnuradio") is not None


def _unitaria(x):
    return x / np.sqrt(max(potencia(x), 1e-30))


def sintetizar(puntos, generador, canal, cfg, rng):
    """Devuelve iq, sin_ruido, ruido y los parámetros sorteados.

    `puntos` es una constelación (camino original, sin cambios) o una `Modulacion` del catálogo digital.
    """
    tx, medio = GENERADORES[generador], TODOS_CANALES[canal]
    parametros = {}
    if isinstance(puntos, Modulacion):
        x, parametros["modulacion"] = puntos.transmitir(generador, cfg, cfg.n_simbolos + 2 * GUARDA, rng)
        x = _unitaria(np.asarray(x, dtype=np.complex128))
    else:
        puntos = np.asarray(puntos, dtype=np.complex128)
        simbolos = puntos[rng.integers(0, len(puntos), cfg.n_simbolos + 2 * GUARDA)]
        x = _unitaria(np.asarray(tx.crear(simbolos, cfg, rng), dtype=np.complex128))
    if medio.antes is not None:
        x, parametros["antes"] = medio.antes(x, cfg.sps, rng)
    inicio, n = round(GUARDA * cfg.sps), round(cfg.n_simbolos * cfg.sps)
    y = np.asarray(x)[inicio:inicio + n]
    if len(y) != n:
        raise ValueError(f"{generador}/{canal} entregó {len(y)} muestras útiles y se esperaban {n}.")
    fase = float(rng.uniform(-np.pi, np.pi))
    y = _unitaria(y) * np.exp(1j * (fase + 2 * np.pi * cfg.cfo * np.arange(n)))
    sigma = np.sqrt(10 ** (-cfg.snr_db / 10) / 2)
    ruido = sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))
    z = y + ruido
    if medio.despues is not None:
        z, parametros["despues"] = medio.despues(z, cfg.sps, rng)
    ganancia = float(rng.uniform(0.5, 1.5))
    parametros.update(fase=fase, ganancia=ganancia)
    return {"iq": (z * ganancia).astype(np.complex64), "sin_ruido": y, "ruido": ruido, "parametros": parametros}
