"""Regulador de ISI: exactitud contra la relación trayecto principal / ecos, y comparación de entrenamientos.

``curva --etiqueta X`` corre el clasificador integrado actual (``results/familias``) sobre:

- ``data/banco/digitales_isi``: los 14 canales con ecos, 6 niveles fijos de ISI (0-30 dB), SNR >= 10 dB;
- la partición de evaluación de ``data/banco/digitales`` (sin regulador, el mismo banco de siempre), por canal.

y escribe ``results/familias_isi/curva_<X>.json``. ``comparar`` junta todas las curvas en ``comparacion.json``.

Orden usado (2026-11-27):
    1. curva --etiqueta natural           (modelos entrenados con digitales_entreno sin regulador)
    2. respaldo results/familias -> results/familias_natural; digitales_entreno_isi pasa a ser digitales_entreno
    3. reentrenar: exp_f1 --etapa final, exp_f3 --etapa simbolos/entrenar, exp_f2 --etapa todo
    4. curva --etiqueta graduada; comparar

    .venv/Scripts/python.exe -m experiments.exp_f6_isi curva --etiqueta natural
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import exp_f2_integracion as f2
from src.banco.canales_extra import CANALES_CON_ECOS
from src.cascada.decisor import DESCONOCIDA

# ─────────────────────────────  Config  ──────────────────────────────
SALIDA = Path("results/familias_isi")


# ───────────────────────────  Predicciones  ──────────────────────────
def predicciones(nombre, trabajos=12):
    """Salida del clasificador integrado actual (con umbral) y su clase más probable, para un banco no de entreno."""
    paquete = joblib.load(f2.PAQUETE_INTEGRADO)
    decisor = paquete["decisor"]
    d = f2.con_especialista(f2.huellas(nombre, trabajos), nombre)
    d = f2._ajuste(d, trabajos, paquete["consistencia"], nombre)[0]
    orden, p = f2.fusion_lote(decisor, d, f2._p_decisor(decisor, d), paquete["con_tonos"])
    return d, f2._salidas(p, orden, decisor.umbral), np.array(orden)[p.argmax(1)]


def _exactitud(d, salida, argmax, m):
    conocida = m & (d["etiqueta"] != DESCONOCIDA)
    if not conocida.any():
        return None
    return {"n": int(conocida.sum()), "exactitud": round(float(np.mean(salida[conocida] == d["etiqueta"][conocida])), 4),
            "sin_umbral": round(float(np.mean(argmax[conocida] == d["etiqueta"][conocida])), 4)}


# ─────────────────────────────  Curva  ───────────────────────────────
def curva(etiqueta, trabajos=12):
    d, salida, argmax = predicciones("isi", trabajos)
    fila = {f["archivo"]: f for f in f2.manifiesto(f2.BANCOS["isi"])}
    nivel = np.array([fila[a]["isi_db"] for a in d["archivo"]], float)
    niveles = sorted(set(nivel.tolist()))
    resultado = {"etiqueta": etiqueta, "niveles": niveles,
                 "isi": {str(n): _exactitud(d, salida, argmax, nivel == n) for n in niveles},
                 "isi_por_canal": {c: {str(n): _exactitud(d, salida, argmax, (nivel == n) & (d["canal"] == c))
                                       for n in niveles} for c in sorted(set(d["canal"]))}}
    e, salida_e, argmax_e = predicciones("evaluacion", trabajos)
    _, ev = f2._particion(e)
    alta = ev & (e["snr"] >= f2.SNR_OPERACION)
    ecos = np.isin(e["canal"], CANALES_CON_ECOS)
    resultado["evaluacion"] = {"todas": _exactitud(e, salida_e, argmax_e, alta),
                               "con_ecos": _exactitud(e, salida_e, argmax_e, alta & ecos),
                               "sin_ecos": _exactitud(e, salida_e, argmax_e, alta & ~ecos),
                               "por_canal": {c: _exactitud(e, salida_e, argmax_e, alta & (e["canal"] == c))
                                             for c in sorted(set(e["canal"]))},
                               "desconocidas_detectadas": round(float(np.mean(
                                   salida_e[alta & (e["etiqueta"] == DESCONOCIDA)] == DESCONOCIDA)), 4)}
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / f"curva_{etiqueta}.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    print(etiqueta, "isi", {n: v["exactitud"] for n, v in resultado["isi"].items()}, flush=True)
    print(etiqueta, "evaluación", {k: v["exactitud"] for k, v in resultado["evaluacion"].items()
                                   if isinstance(v, dict) and "exactitud" in v}, flush=True)


def comparar():
    curvas = {c["etiqueta"]: c for c in (json.loads(p.read_text("utf-8")) for p in sorted(SALIDA.glob("curva_*.json")))}
    (SALIDA / "comparacion.json").write_text(json.dumps(curvas, indent=1, ensure_ascii=False), encoding="utf-8")
    for nombre, c in curvas.items():
        print(nombre, "isi", {n: v["exactitud"] for n, v in c["isi"].items()},
              "evaluación", {k: c["evaluacion"][k]["exactitud"] for k in ("todas", "con_ecos", "sin_ecos")},
              "desconocidas", c["evaluacion"]["desconocidas_detectadas"], flush=True)


# ─────────────────────────────  Entry  ───────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=("curva", "comparar"))
    p.add_argument("--etiqueta", default="actual")
    p.add_argument("--trabajos", type=int, default=12)
    a = p.parse_args()
    curva(a.etiqueta, a.trabajos) if a.accion == "curva" else comparar()


if __name__ == "__main__":
    main()
