"""Mide en el banco la cadena con seguimiento y la votación por tramos frente a la cadena simple.

    .venv/Scripts/python.exe -m experiments.exp_c9_seguimiento --paquete results/cascada/modelos_robustos/paquete_imagen.joblib
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from experiments.comun_cascada import fila_resultado, metricas
from src.cascada import geometria as g
from src.cascada.pipeline import analizar_cascada, cargar_paquete
from src.cascada.tramos import analizar_por_tramos

# ─────────────────────────────  Config  ──────────────────────────────
BANCOS = {"base": Path("data/banco/base"), "canales2": Path("data/banco/canales2")}
NOMBRES = list(g.catalogo())
MODOS = ("red", "mixta_seguimiento", "red_tramos")
SALIDA = Path("results/cascada/seguimiento")
POR_BANCO = 1200
_C = {}


# ─────────────────────────────  Análisis  ────────────────────────────
def paquete(ruta):
    if _C.get("ruta") != ruta:
        torch.set_num_threads(1)
        _C.update(ruta=ruta, p=cargar_paquete(ruta))
    return _C["p"]


def analizar(ruta, base, f):
    x = np.fromfile(base / f["archivo"], np.complex64)
    p = paquete(ruta)
    salida = {}
    for modo in MODOS:
        cadena = "red" if modo == "red_tramos" else modo
        q = {**p, "cadena": cadena}
        inicio = time.time()
        r = analizar_por_tramos(x, q, largo=16384) if modo == "red_tramos" else analizar_cascada(x, q)
        salida[modo] = ({k: r.get(k) for k in ("modulacion", "confianza", "probabilidades", "etapas")},
                        time.time() - inicio)
    return salida


def muestra(semilla):
    rng = np.random.default_rng(semilla)
    elegidos = []
    for nombre, base in BANCOS.items():
        filas = [json.loads(x) for x in (base / "manifiesto.jsonl").read_text(encoding="utf-8").splitlines()]
        filas = [f for f in filas if f["grupo"] != "validacion" and f["snr_db"] >= 10]
        for i in rng.choice(len(filas), min(POR_BANCO, len(filas)), replace=False):
            elegidos.append((base, {**filas[i], "banco": nombre}))
    return elegidos


def resumen(filas):
    m = metricas(filas, NOMBRES)
    return {"archivos": m["n"], "exactitud": m["exactitud_cerrada"], "correctas_salida_real": m["exactitud_abierta"],
            "desconocidas_detectadas": m["rechazo_desconocidas"], "auroc": m["auroc_desconocidas"]}


def por_canal(filas):
    grupos = {}
    for f in filas:
        if f["etiqueta"] in NOMBRES:
            grupos.setdefault(f["canal"], []).append(f["cerrada"] == f["etiqueta"])
    return {k: round(float(np.mean(v)), 3) for k, v in sorted(grupos.items())}


# ──────────────────────────────  Entry  ──────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paquete", required=True)
    p.add_argument("--trabajos", type=int, default=10)
    a = p.parse_args()
    elegidos = muestra(2026112430)
    salidas = Parallel(n_jobs=a.trabajos, batch_size=4)(delayed(analizar)(a.paquete, b, f) for b, f in elegidos)
    total = {}
    for modo in MODOS:
        filas = []
        for (_, f), s in zip(elegidos, salidas):
            item = {"real": f["modulacion"], "etiqueta": f["etiqueta"], "snr": f["snr_db"], "sps": f["sps"]}
            filas.append({**fila_resultado(item, s[modo][0], NOMBRES), "canal": f["canal"], "banco": f["banco"]})
        total[modo] = {"todo": resumen(filas), "segundos_medios": float(np.mean([s[modo][1] for s in salidas])),
                       **{b: resumen([x for x in filas if x["banco"] == b]) for b in BANCOS},
                       "por_canal": por_canal(filas)}
    SALIDA.mkdir(parents=True, exist_ok=True)
    nombre = Path(a.paquete).stem
    (SALIDA / f"{nombre}.json").write_text(json.dumps(total, indent=1), encoding="utf-8")
    print(json.dumps({m: {"todo": v["todo"], "segundos": round(v["segundos_medios"], 2)} for m, v in total.items()},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
