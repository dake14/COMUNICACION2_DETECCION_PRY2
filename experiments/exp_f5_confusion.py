"""Matrices de confusión de todas las configuraciones, sobre la partición de evaluación de ``digitales``.

Filas: la modulación real (las 18 conocidas y cada desconocida por separado). Columnas: lo que respondió cada
configuración (sus clases, «DESCONOCIDA» e «INDETERMINADA»). Escribe ``results/familias/confusion/confusion.json``,
que sirve la interfaz en ``/api/confusion``.

Las cascadas viejas (red, mixta) no tienen huellas de este banco: se corren archivo por archivo (caché en
``results/familias/confusion/cascada_<id>.npz``). Las configuraciones integradas usan el paquete actual; la etapa 2b y la
4 se reconstruyen quitándole el especialista y la prueba de ajuste, con la misma calibración.

    .venv/Scripts/python.exe -m experiments.exp_f5_confusion
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import datetime
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed

from experiments import exp_f2_integracion as f2
from src.banco.modulaciones import catalogo_digital, desconocidas_prueba_digital
from src.cascada.decisor import DESCONOCIDA, INDETERMINADA
from src.cascada.pipeline import analizar_cascada, cargar_paquete

# ─────────────────────────────  Config  ──────────────────────────────
SALIDA = Path("results/familias/confusion")
CASCADAS = {"red": Path("results/cascada/modelos/paquete_red.joblib"),
            "mixta": Path("results/cascada/modelos/paquete.joblib")}
GRUPOS_SNR = {"todas": lambda s: np.ones(len(s), bool), "snr>=10": lambda s: s >= f2.SNR_OPERACION,
              "snr<10": lambda s: s < f2.SNR_OPERACION}
CONFIGURACIONES = [
    ("red", "Cascada «Red + refino»", "Primera versión (2026-11-18). Solo conoce 6 clases (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 4PAM): las otras 12 nunca puede acertarlas.", "cascada"),
    ("mixta", "Cascada «Mixta + refino»", "Primera versión con tres rutas de recuperación. Solo conoce las mismas 6 clases.", "cascada"),
    ("robusto", "Cascada robusta", "32 canales, recuperador nuevo y CNN adversaria de dominio (2026-11-24). Solo conoce las mismas 6 clases.",
     "cascada"),
    ("enrutador", "Enrutador de familias solo", "Etapa 2: 18 clases, sin constelación ni «desconocida».", "enrutador"),
    ("fusion", "Enrutador × decisor", "Etapa 2b: la constelación afina dentro de la familia.", "integrada"),
    ("especialista", "+ especialista QAM/APSK", "Etapa 4: histogramas finos de la constelación.", "integrada"),
    ("ajuste", "+ ajuste a la constelación", "Etapa 5 (actual): la clase que no cabe cede masa a «desconocida».",
     "integrada"),
]
_C = {}


# ─────────────────────────────  Cascadas  ────────────────────────────
def _cascada(clave, archivo):
    if clave not in _C:
        torch.set_num_threads(1)
        _C[clave] = cargar_paquete(CASCADAS[clave])
    x = np.fromfile(f2.BANCOS["evaluacion"] / archivo, np.complex64)
    try:
        return analizar_cascada(x, _C[clave]).get("modulacion") or INDETERMINADA
    except Exception:  # una señal que rompe la cascada vieja cuenta como sin decidir
        return INDETERMINADA


def predicciones_cascada(clave, archivos, trabajos=12):
    cache = SALIDA / f"cascada_{clave}.npz"
    if cache.exists():
        d = np.load(cache)
        assert (d["archivo"] == archivos).all()
        return d["prediccion"]
    inicio = time.time()
    pred = np.array(Parallel(n_jobs=trabajos, batch_size=8)(delayed(_cascada)(clave, a) for a in archivos))
    SALIDA.mkdir(parents=True, exist_ok=True)
    np.savez(cache, archivo=archivos, prediccion=pred)
    print("cascada", clave, len(archivos), f"{(time.time() - inicio) / 60:.1f} min", flush=True)
    return pred


# ───────────────────────────  Integradas  ────────────────────────────
def predicciones_integradas(d, trabajos):
    paquete = joblib.load(f2.PAQUETE_INTEGRADO)
    decisor = paquete["decisor"]
    p_dec = f2._p_decisor(decisor, d)
    sin_esp = {k: v for k, v in d.items() if k not in ("p_especialista", "clases_especialista")}
    con_ajuste = f2._ajuste(d, trabajos, paquete["consistencia"])[0]
    salida = {}
    for clave, datos in (("fusion", sin_esp), ("especialista", d), ("ajuste", con_ajuste)):
        orden, p = f2.fusion_lote(decisor, datos, p_dec, paquete["con_tonos"])
        salida[clave] = (f2._salidas(p, orden, decisor.umbral), np.array(orden)[p.argmax(1)], orden)
    robusto = joblib.load(f2.PAQUETE_ROBUSTO)["decisor"]
    p_rob = f2._p_decisor(robusto, d)
    salida["robusto"] = (np.where(d["ok"], f2._salidas(np.nan_to_num(p_rob), robusto.nombres, robusto.umbral),
                                  INDETERMINADA), None, list(robusto.nombres))
    clases = list(d["clases_enrutador"])
    salida["enrutador"] = (np.array(clases)[d["p_enrutador"].argmax(1)], None, clases)
    return salida, float(decisor.umbral)


# ─────────────────────────────  Matrices  ────────────────────────────
def _columnas(clases):
    return [c for c in clases if c not in (DESCONOCIDA, INDETERMINADA)] + [DESCONOCIDA, INDETERMINADA]


def _matrices(real, pred, snr, filas, columnas):
    salida = {}
    for grupo, criterio in GRUPOS_SNR.items():
        m = criterio(snr)
        cuenta = np.zeros((len(filas), len(columnas)), int)
        for r, p in zip(real[m], pred[m]):
            cuenta[filas.index(r), columnas.index(p)] += 1
        salida[grupo] = cuenta.tolist()
    return salida


def construir(trabajos=12):
    d = f2.con_especialista(f2.huellas("evaluacion", trabajos))
    _, ev = f2._particion(d)
    integradas, umbral = predicciones_integradas(d, trabajos)
    integradas = {k: (v[0][ev], None if v[1] is None else v[1][ev], v[2]) for k, v in integradas.items()}
    d = f2._sub(d, ev)
    real, snr = d["modulacion"], d["snr"]
    conocidas = list(catalogo_digital())
    filas = conocidas + [m for m in desconocidas_prueba_digital() if m in set(real)]
    familia_de = {**{k: m.familia for k, m in catalogo_digital().items()},
                  **{k: "desconocida" for k in filas if k not in conocidas}}
    configuraciones = []
    for clave, nombre, descripcion, tipo in CONFIGURACIONES:
        if tipo == "cascada" and clave in CASCADAS:
            pred, argmax = predicciones_cascada(clave, d["archivo"], trabajos), None
            clases = list(cargar_paquete(CASCADAS[clave])["decisor"].nombres)
        else:
            pred, argmax, clases = integradas[clave]
        columnas = _columnas(clases)
        assert set(pred) <= set(columnas), set(pred) - set(columnas)
        configuraciones.append({
            "id": clave, "nombre": nombre, "descripcion": descripcion, "columnas": columnas,
            "matrices": _matrices(real, pred, snr, filas, columnas),
            "sin_umbral": None if argmax is None else {"columnas": columnas,
                                                       "matrices": _matrices(real, argmax, snr, filas, columnas)}})
        print(clave, "listo", flush=True)
    resultado = {"particion": "evaluación de digitales: catálogo con índice 1 y desconocidas de prueba",
                 "archivos": int(len(real)), "generado": datetime.date.today().isoformat(),
                 "umbral_integrado": round(umbral, 3), "filas": filas, "familia_de": familia_de,
                 "configuraciones": configuraciones}
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "confusion.json").write_text(json.dumps(resultado, ensure_ascii=False), encoding="utf-8")
    print("guardado", SALIDA / "confusion.json", flush=True)


# ─────────────────────────────  Entry  ───────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trabajos", type=int, default=12)
    construir(p.parse_args().trabajos)


if __name__ == "__main__":
    main()
