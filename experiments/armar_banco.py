"""Genera el banco cruzando generadores, canales, modulaciones y configuraciones.

Con los bloques de GNU Radio:
    C:/Users/David/radioconda/python.exe -m experiments.armar_banco --salida data/banco/base --por-celda 2

Catálogo de modulaciones digitales básicas (18 clases, FSK/MSK/GMSK incluidas):
    C:/Users/David/radioconda/python.exe -m experiments.armar_banco --catalogo digital --salida data/banco/digitales

Regulador de ISI (relación trayecto principal / ecos en dB, ``src/banco/isi.py``), solo en ``CANALES_CON_ECOS``:
``--isi-aleatorio 0.5,0,30`` le da a la mitad de esos archivos un nivel U(0, 30) dB y deja la otra mitad con los ecos
sorteados; ``--isi-niveles 0,5,10`` asigna los niveles en ciclo por índice. El nivel sale de un generador aparte: el
resto del archivo (símbolos, ruido, retardos) es idéntico al del mismo banco sin regulador.
"""
import argparse
import dataclasses
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.banco import CANALES, GENERADORES, gnuradio_disponible, muestrear_config, sintetizar
from src.banco.canales_extra import CANALES_CON_ECOS, TODOS_CANALES
from src.banco.config import N_SIMBOLOS, SNRS
from src.banco.modulaciones import modulaciones_digitales
from src.cascada import geometria as g

SEMILLA = 2026112001


def modulaciones(catalogo="original"):
    if catalogo == "digital":
        return modulaciones_digitales()
    salida = {k: (v, k, "catalogo") for k, v in g.catalogo().items()}
    salida.update({k: (v, "DESCONOCIDA", "validacion") for k, v in g.desconocidas_validacion().items()})
    salida.update({k: (v, "DESCONOCIDA", "prueba") for k, v in g.desconocidas_prueba().items()})
    return salida


def disponibles(registro):
    return [k for k, v in registro.items() if gnuradio_disponible() or not v.gnuradio]


def _nivel_isi(isi, canal, semilla, clave, i):
    if isi is None or canal not in CANALES_CON_ECOS:
        return None
    if isi[0] == "niveles":
        return float(isi[1][i % len(isi[1])])
    _, fraccion, bajo, alto = isi
    r = np.random.default_rng([semilla, clave, 7])
    return float(r.uniform(bajo, alto)) if r.uniform() < fraccion else None


def _tarea(t):
    salida, generador, canal, modulacion, i, semilla, n_simbolos, snrs, extremos, catalogo, isi = t
    puntos, etiqueta, grupo = modulaciones(catalogo)[modulacion]
    clave = zlib.crc32(f"{generador}|{canal}|{modulacion}|{i}".encode())
    rng = np.random.default_rng([semilla, clave])
    cfg = muestrear_config(generador, rng, n_simbolos, snrs, extremos)
    cfg = dataclasses.replace(cfg, isi_db=_nivel_isi(isi, canal, semilla, clave, i))
    r = sintetizar(puntos, generador, canal, cfg, rng)
    relativo = Path(generador) / canal / f"{modulacion}_{i:03d}.c64"
    destino = Path(salida) / relativo
    destino.parent.mkdir(parents=True, exist_ok=True)
    r["iq"].tofile(destino)
    return {"archivo": relativo.as_posix(), "muestras": len(r["iq"]), "generador": generador, "canal": canal,
            "modulacion": modulacion, "etiqueta": etiqueta, "grupo": grupo, "indice": i,
            **({"familia": puntos.familia} if catalogo == "digital" else {}),
            **cfg.como_dict(), "parametros_canal": r["parametros"]}


def armar(salida, generadores, canales, nombres, por_celda=1, semilla=SEMILLA, trabajos=1,
          n_simbolos=N_SIMBOLOS, snrs=SNRS, extremos=0.0, catalogo="original", isi=None):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    tareas = [(str(salida), a, b, m, i, semilla, tuple(n_simbolos), tuple(snrs), extremos, catalogo, isi)
              for a in generadores for b in canales for m in nombres for i in range(por_celda)]
    if trabajos > 1:
        with ProcessPoolExecutor(trabajos) as grupo:
            filas = list(grupo.map(_tarea, tareas, chunksize=4))
    else:
        filas = [_tarea(t) for t in tareas]
    texto = "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas)
    (salida / "manifiesto.jsonl").write_text(texto, encoding="utf-8")
    return filas


def _lista(texto, validos, tipo):
    if texto is None:
        return list(validos)
    elegidos = texto.split(",")
    faltan = sorted(set(elegidos) - set(validos))
    if faltan:
        raise SystemExit(f"{tipo} no disponibles: {', '.join(faltan)}")
    return elegidos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salida", default="data/banco/base")
    p.add_argument("--generadores")
    p.add_argument("--canales")
    p.add_argument("--modulaciones")
    p.add_argument("--por-celda", type=int, default=1)
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--trabajos", type=int, default=2)
    p.add_argument("--extremos", type=float, default=0.0)
    p.add_argument("--catalogo", choices=("original", "digital"), default="original")
    p.add_argument("--snrs", help="Lista de SNR en dB separada por comas (por defecto las de config.SNRS).")
    regulador = p.add_mutually_exclusive_group()
    regulador.add_argument("--isi-aleatorio", help="fraccion,bajo,alto en dB, p. ej. 0.5,0,30")
    regulador.add_argument("--isi-niveles", help="niveles en dB separados por comas, en ciclo por índice")
    a = p.parse_args()
    isi = None
    if a.isi_aleatorio:
        fraccion, bajo, alto = (float(v) for v in a.isi_aleatorio.split(","))
        isi = ("aleatorio", fraccion, bajo, alto)
    elif a.isi_niveles:
        isi = ("niveles", tuple(float(v) for v in a.isi_niveles.split(",")))
    snrs = tuple(float(v) for v in a.snrs.split(",")) if a.snrs else SNRS
    filas = armar(a.salida, _lista(a.generadores, disponibles(GENERADORES), "Generadores"),
                  _lista(a.canales, disponibles(TODOS_CANALES if a.canales or a.catalogo == "digital" else CANALES), "Canales"), _lista(a.modulaciones, modulaciones(a.catalogo), "Modulaciones"),
                  a.por_celda, a.semilla, a.trabajos, snrs=snrs, extremos=a.extremos, catalogo=a.catalogo, isi=isi)
    print(len(filas), "archivos en", a.salida)


if __name__ == "__main__":
    main()
