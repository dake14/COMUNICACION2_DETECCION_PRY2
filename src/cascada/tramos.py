"""Análisis por tramos con votación de probabilidades.

CONTRATO (unidad U4, cerrado):

Dependencias permitidas (lista cerrada): math, numpy, src.cascada.pipeline.analizar_cascada,
src.cascada.decisor.INDETERMINADA.

- ``analizar_por_tramos(iq, paquete, sps=None, largo=32768, maximo=4) -> dict``
  1. ``x = np.asarray(iq, dtype=np.complex128).ravel()``.
  2. Si ``len(x) <= largo`` o ``maximo <= 1``: ``r = analizar_cascada(x[:largo] if maximo <= 1 else x, paquete, sps)``;
     devolver ``{**r, "tramos": [_resumen(r, 0)]}``.
  3. ``n = min(maximo, ceil(len(x) / largo))``; inicios ``np.linspace(0, len(x) - largo, n).astype(int)`` (únicos, en orden).
  4. Primer tramo: ``r0 = analizar_cascada(x[i0:i0 + largo], paquete, sps)``. Tasa para el resto:
     ``sps`` si no es None; si no, ``r0["etapas"]["sps_origen"]`` cuando ``r0`` tiene la clave "etapas"; si no, None.
     Los demás tramos se analizan con esa tasa.
  5. Válidos: resultados con ``estado == "clasificada"`` y clave "probabilidades". Si no hay válidos, devolver
     ``{**r0, "tramos": [...]}``.
  6. Combinación: para cada nombre en ``paquete["decisor"].nombres`` (en ese orden), media sobre los válidos de
     ``log(max(p, 1e-12))``; luego softmax estable (restar el máximo). ``k = argmax``; ``confianza = p[k]``;
     ``modulacion = nombres[k]`` salvo que ``confianza < paquete["decisor"].umbral``, en cuyo caso ``INDETERMINADA``.
  7. Devolver una copia del resultado válido de mayor "confianza" con las claves "modulacion", "confianza" (float),
     "probabilidades" (dict nombre -> float) reemplazadas por las combinadas, y "tramos": lista con un
     ``_resumen(r, inicio)`` por tramo analizado, en orden de inicio.
- ``_resumen(r, inicio) -> dict``: ``{"inicio": int(inicio), "modulacion": r.get("modulacion"),
  "confianza": None si r.get("confianza") es None, si no float(...)}``.

Criterio de aceptación: con un paquete falso cuyo ``analizar_cascada`` se sustituye en las pruebas
(monkeypatch de ``src.cascada.tramos.analizar_cascada``), tres tramos con probabilidades {A: .6, B: .3, D: .1},
{A: .5, B: .4, D: .1} y {A: .2, B: .7, D: .1} y umbral 0.4 dan modulación A o B según la media logarítmica
calculada a mano; con un solo tramo devuelve el mismo resultado que analizar_cascada más "tramos".
"""
import math

import numpy as np

from src.cascada.decisor import INDETERMINADA
from src.cascada.pipeline import analizar_cascada


def _resumen(r, inicio):
    confianza = r.get("confianza")
    return {
        "inicio": int(inicio),
        "modulacion": r.get("modulacion"),
        "confianza": None if confianza is None else float(confianza),
    }


def analizar_por_tramos(iq, paquete, sps=None, largo=32768, maximo=4):
    x = np.asarray(iq, dtype=np.complex128).ravel()
    if len(x) <= largo or maximo <= 1:
        r = analizar_cascada(x[:largo] if maximo <= 1 else x, paquete, sps)
        return {**r, "tramos": [_resumen(r, 0)]}

    n = min(maximo, math.ceil(len(x) / largo))
    inicios = list(dict.fromkeys(int(i) for i in np.linspace(0, len(x) - largo, n).astype(int)))

    i0 = inicios[0]
    r0 = analizar_cascada(x[i0:i0 + largo], paquete, sps)
    tasa = sps if sps is not None else (r0.get("etapas") or {}).get("sps_origen")

    resultados = [r0]
    for inicio in inicios[1:]:
        resultados.append(analizar_cascada(x[inicio:inicio + largo], paquete, tasa))

    tramos = [_resumen(r, inicio) for inicio, r in zip(inicios, resultados)]
    validos = [r for r in resultados if r.get("estado") == "clasificada" and "probabilidades" in r]
    if not validos:
        return {**r0, "tramos": tramos}

    decisor = paquete["decisor"]
    nombres = list(decisor.nombres)
    log_medias = np.array(
        [
            sum(math.log(max(float(r["probabilidades"].get(nombre, 0.0)), 1e-12)) for r in validos) / len(validos)
            for nombre in nombres
        ],
        dtype=float,
    )
    exp_medias = np.exp(log_medias - np.max(log_medias))
    probs = exp_medias / np.sum(exp_medias)
    k = int(np.argmax(probs))
    confianza = float(probs[k])
    modulacion = INDETERMINADA if confianza < decisor.umbral else nombres[k]

    mejor = max(validos, key=lambda r: -math.inf if r.get("confianza") is None else float(r["confianza"]))
    return {
        **mejor,
        "modulacion": modulacion,
        "confianza": confianza,
        "probabilidades": {nombre: float(p) for nombre, p in zip(nombres, probs)},
        "tramos": tramos,
    }
