"""Recalibra temperatura, puerta de desconocidas y umbral del paquete robusto con archivos del banco que exp_c9 no usa.

Conocidas: archivos `catalogo` fuera de la muestra de exp_c9. Desconocidas: grupo `validacion` (24QAM, 6PSK, ANILLOS12,
HEX19), geometrías distintas de las de prueba. Las mezclas del catálogo y del fondo no se tocan.

- ``completa``: rehace temperatura, puerta y umbral con ``Decisor.calibrar``.
- ``desplazamiento``: conserva la puerta del pool (mismo orden de P(desconocida), mismo AUROC) y solo mueve su
  intercepto. Se elige el mayor desplazamiento que deja como mucho un 10 % de conocidas marcadas como desconocidas en
  la calibración con SNR ≥ 10 dB. (Maximizar la media de conocidas correctas y desconocidas detectadas llevaba a un
  extremo: las conocidas indeterminadas pasaban a «desconocida» sin costo en esa media.)

    .venv/Scripts/python.exe -m experiments.exp_c10_recalibrar --modo desplazamiento
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed

from experiments.exp_c9_seguimiento import BANCOS, muestra
from src.cascada.cadena import recuperar
from src.cascada.decisor import DESCONOCIDA, INDETERMINADA
from src.cascada.detector import snr_ciega_db
from src.cascada.pipeline import calidad_de, cargar_paquete, huella_de_simbolos

# ─────────────────────────────  Config  ──────────────────────────────
ORIGEN = Path("results/cascada/modelos_robustos/paquete_adversaria.joblib")
RESPALDO = ORIGEN.with_name("paquete_adversaria_pool.joblib")
DESTINOS = {"completa": ORIGEN.with_name("paquete_adversaria_recal.joblib"),
            "desplazamiento": ORIGEN.with_name("paquete_adversaria_desplazada.joblib")}
DESPLAZAMIENTOS = np.round(np.arange(-2.0, 2.001, 0.05), 2)
MAX_CONOCIDAS_COMO_DESCONOCIDAS = 0.10
SNR_OPERACION = 10.0
CACHE = Path("results/cascada/seguimiento/calibracion_banco.npz")
SEMILLA_C9 = 2026112430
_C = {}


# ───────────────────────────────  Datos  ─────────────────────────────
def archivos_calibracion():
    usados = {(str(b), f["archivo"]) for b, f in muestra(SEMILLA_C9)}
    elegidos = []
    for base in BANCOS.values():
        for linea in (base / "manifiesto.jsonl").read_text(encoding="utf-8").splitlines():
            f = json.loads(linea)
            if f["grupo"] in ("catalogo", "validacion") and (str(base), f["archivo"]) not in usados:
                elegidos.append((base, f))
    return elegidos


def paquete():
    if "p" not in _C:
        torch.set_num_threads(1)
        _C["p"] = cargar_paquete(ORIGEN)
    return _C["p"]


def huella(base, f):
    x = np.fromfile(base / f["archivo"], np.complex64).astype(np.complex128)
    p = paquete()
    r = recuperar(x, p["cadena"], p.get("recuperador"))
    if r is None:
        return None
    z, _ = huella_de_simbolos(r["simbolos"], p)
    return z[0], calidad_de(r, snr_ciega_db(x))[0]


def extraer(trabajos):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["z"], d["calidad"], d["etiqueta"].tolist(), d["snr"]
    elegidos = archivos_calibracion()
    salidas = Parallel(n_jobs=trabajos, batch_size=4)(delayed(huella)(b, f) for b, f in elegidos)
    pares = [(s, f) for s, (_, f) in zip(salidas, elegidos) if s is not None]
    z = np.stack([s[0] for s, _ in pares])
    calidad = np.stack([s[1] for s, _ in pares])
    etiqueta = [f["etiqueta"] for _, f in pares]
    snr = np.array([f["snr_db"] for _, f in pares], float)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, z=z, calidad=calidad, etiqueta=np.array(etiqueta), snr=snr)
    print("huellas", len(pares), "de", len(elegidos), flush=True)
    return z, calidad, etiqueta, snr


# ───────────────────────────────  Core  ──────────────────────────────
def tasas(decisor, z, calidad, etiqueta):
    p = decisor.probabilidades(z, calidad)
    k = p.argmax(1)
    nombres = np.array(decisor.nombres)
    salida = np.where(p[np.arange(len(p)), k] >= decisor.umbral, nombres[k], INDETERMINADA)
    etiqueta = np.array(etiqueta)
    desconocida = etiqueta == DESCONOCIDA
    return {"conocidas_correctas": float(np.mean(salida[~desconocida] == etiqueta[~desconocida])),
            "conocidas_como_desconocidas": float(np.mean(salida[~desconocida] == DESCONOCIDA)),
            "desconocidas_detectadas": float(np.mean(salida[desconocida] == DESCONOCIDA))}


def desplazar(decisor, z, calidad, etiqueta, snr):
    m = snr >= SNR_OPERACION
    z, calidad, etiqueta = z[m], calidad[m], [e for e, x in zip(etiqueta, m) if x]
    regresion = decisor.puerta[-1]
    base = regresion.intercept_.copy()
    elegido, antes = 0.0, tasas(decisor, z, calidad, etiqueta)
    for d in DESPLAZAMIENTOS:
        regresion.intercept_ = base + d
        if tasas(decisor, z, calidad, etiqueta)["conocidas_como_desconocidas"] <= MAX_CONOCIDAS_COMO_DESCONOCIDAS:
            elegido = float(d)
    regresion.intercept_ = base + elegido
    despues = tasas(decisor, z, calidad, etiqueta)
    print("desplazamiento", elegido, "| antes", {k: round(v, 3) for k, v in antes.items()},
          "| después", {k: round(v, 3) for k, v in despues.items()}, flush=True)
    return elegido


def recalibrar(modo, trabajos=12):
    z, calidad, etiqueta, snr = extraer(trabajos)
    datos = joblib.load(ORIGEN)
    decisor = copy.deepcopy(datos["decisor"])
    anterior = {"temperatura": decisor.temperatura, "umbral": decisor.umbral}
    if modo == "completa":
        decisor.calibrar(z, etiqueta, calidad)
        detalle = "banco: catalogo fuera de exp_c9 + validacion"
    else:
        detalle = f"puerta del pool desplazada {desplazar(decisor, z, calidad, etiqueta, snr):+.2f} con el banco"
    print("antes", anterior, "después", {"temperatura": decisor.temperatura, "umbral": decisor.umbral},
          "conocidas", sum(e != DESCONOCIDA for e in etiqueta), "desconocidas", etiqueta.count(DESCONOCIDA),
          flush=True)
    if not RESPALDO.exists():
        shutil.copyfile(ORIGEN, RESPALDO)
    joblib.dump({**datos, "decisor": decisor, "metadatos": {**datos["metadatos"], "calibracion": detalle}},
                DESTINOS[modo], compress=3)
    print("guardado", DESTINOS[modo], flush=True)


# ─────────────────────────────  Entry  ───────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modo", choices=tuple(DESTINOS), default="desplazamiento")
    p.add_argument("--trabajos", type=int, default=12)
    a = p.parse_args()
    recalibrar(a.modo, a.trabajos)


if __name__ == "__main__":
    main()
