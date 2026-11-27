"""Genera señales de entrenamiento con todos los generadores y canales del banco.

Ejecutar con el Python de radioconda para incluir los bloques de GNU Radio.
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import json
import math
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.banco import CANALES, GENERADORES, muestrear_config, sintetizar
from src.banco.canales_extra import TODOS_CANALES
from src.cascada import geometria as g

# ─────────────────────────────  Config  ──────────────────────────────
SEMILLA = 2026112010
DESTINO = Path("data/cascada_multi")
GEOMETRIAS = 3000
VISTAS = 6
POR_CATALOGO = 1200
CALIBRA_CATALOGO = 400
CALIBRA_VALIDACION = 200
CALIBRA_ALEATORIAS = 1200
N_SIMBOLOS = (256, 512, 1024)
TRABAJOS = 10
SEMILLA_VARIADA = 2026112402
EXTREMOS = 0.2


# ───────────────────────────────  Plan  ──────────────────────────────
def plan(escala, variado=False):
    tareas = [("geo", i, None, j) for i in range(GEOMETRIAS // escala) for j in range(VISTAS)]
    for k, nombre in enumerate(g.catalogo()):
        tareas += [("catalogo", k, nombre, j) for j in range(POR_CATALOGO // escala)]
        tareas += [("calibra", k, nombre, j) for j in range(CALIBRA_CATALOGO // escala)]
    for k, nombre in enumerate(g.desconocidas_validacion()):
        tareas += [("calibra", 100 + k, nombre, j) for j in range(CALIBRA_VALIDACION // escala)]
    tareas += [("calibra_geo", i, None, 0) for i in range(CALIBRA_ALEATORIAS // escala)]
    return [(*t, variado) for t in tareas]


# ─────────────────────────────  Señales  ─────────────────────────────
@lru_cache(maxsize=512)
def geometria(tipo, indice, semilla=SEMILLA):
    return g.geometria_aleatoria(np.random.default_rng([semilla, zlib.crc32(tipo.encode()), indice]))


def constelaciones():
    return {**g.catalogo(), **g.desconocidas_validacion()}


def crear_vista(tarea):
    tipo, indice, nombre, vista, variado = tarea
    semilla = SEMILLA_VARIADA if variado else SEMILLA
    puntos = geometria(tipo, indice, semilla) if nombre is None else constelaciones()[nombre]
    rng = np.random.default_rng([semilla, 7, zlib.crc32(tipo.encode()), indice, vista])
    generador = str(rng.choice(list(GENERADORES)))
    canal = str(rng.choice(list(TODOS_CANALES if variado else CANALES)))
    cfg = muestrear_config(generador, rng, N_SIMBOLOS, extremos=EXTREMOS if variado else 0.0)
    iq = sintetizar(puntos, generador, canal, cfg, rng)["iq"]
    etiqueta = nombre if nombre in g.catalogo() else "DESCONOCIDA"
    meta = {"tipo": tipo, "indice": indice, "vista": vista, "modulacion": nombre, "etiqueta": etiqueta,
            "generador": generador, "canal": canal, "sps": cfg.sps, "snr_db": cfg.snr_db,
            "puntos": math.log2(len(puntos)) / 7.0, "anillos": min(g.contar_anillos(puntos), 6) / 6.0}
    return iq, meta


# ─────────────────────────────  Salida  ──────────────────────────────
def generar(destino, escala, trabajos, variado=False):
    destino.mkdir(parents=True, exist_ok=True)
    tareas = plan(escala, variado)
    filas, posicion = [], 0
    with open(destino / "iq.bin", "wb") as binario, ProcessPoolExecutor(trabajos) as grupo:
        for iq, meta in grupo.map(crear_vista, tareas, chunksize=24):
            binario.write(iq.tobytes())
            filas.append({**meta, "offset": posicion, "n": len(iq)})
            posicion += iq.nbytes
    texto = "".join(json.dumps(f) + "\n" for f in filas)
    (destino / "meta.jsonl").write_text(texto, encoding="utf-8")
    print(len(filas), "vistas,", round(posicion / 1e9, 2), "GB", flush=True)


# ──────────────────────────────  Entry  ──────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--variado", action="store_true")
    a = p.parse_args()
    destino = Path("data/cascada_variado") if a.variado else DESTINO
    destino = destino.with_name(destino.name + "_rapido") if a.rapido else destino
    if (destino / "meta.jsonl").exists():
        print("ya existe", destino, flush=True)
        return
    generar(destino, 40 if a.rapido else 1, 4 if a.rapido else TRABAJOS, a.variado)


if __name__ == "__main__":
    main()
