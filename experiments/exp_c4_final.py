"""H5: evaluación final, una sola ejecución, con generadores no usados."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from experiments.comun_cascada import (RUTA_PAQUETE, canales_config, conjunto_repo, fila_resultado,
                                       metricas, transmitir_repo)
from experiments.exp_c3_inscripcion import inscribir_modulacion
from src.cascada import geometria as g
from src.cascada.decisor import DESCONOCIDA
from src.cascada.pipeline import analizar_cascada, cargar_paquete
from src.conventional import make_feature_classifier
from src.features import extract_feature_matrix

PRUEBA = Path("data/cascada_prueba")
SALIDA = Path("results/cascada/final")
NOMBRES = list(g.catalogo())
_P = {}


def cargar_items(generador):
    base = PRUEBA / generador
    items = []
    for m in json.loads((base / "manifiesto.json").read_text(encoding="utf-8")):
        items.append({**m, "iq": np.fromfile(base / m["archivo"], np.complex64)})
    return items


def analizar(item, ruta=RUTA_PAQUETE):
    if "p" not in _P:
        torch.set_num_threads(1)
        _P["p"] = cargar_paquete(ruta)
    inicio = time.perf_counter()
    r = analizar_cascada(item["iq"], _P["p"])
    r["tiempo_s"] = time.perf_counter() - inicio
    return r


def fila_final(item, r):
    deteccion = r.get("deteccion") or {}
    return {**fila_resultado(item, r, NOMBRES), "n_simbolos": item["n_simbolos"], "tiempo_s": r["tiempo_s"],
            "detectada": deteccion.get("detectada"), "archivo": item["archivo"]}


def entrenar_svm(semilla=2026111840):
    rng = np.random.default_rng(semilla)
    canales = [None, *canales_config("id_channels")[1:]]
    x, y = [], []
    for i, (nombre, puntos) in enumerate(g.catalogo().items()):
        for snr in (0, 5, 10, 15, 20, 30):
            for _ in range(130):
                taps = canales[int(rng.integers(len(canales)))]
                iq = transmitir_repo(puntos, 64, rng, snr, 8, float(rng.choice([0.25, 0.35, 0.5])), taps,
                                     float(rng.uniform(-0.0125, 0.0125)))
                x.append(iq[:512])
                y.append(i)
    return make_feature_classifier({"family": "svm_rbf", "feature_set": "full", "C": 10.0}, semilla).fit(
        extract_feature_matrix(np.stack(x)), np.array(y))


def svm_archivo(modelo, item):
    iq = item["iq"]
    ventanas = [iq[i:i + 512] for i in range(0, len(iq) - 511, 512)]
    if not ventanas:
        return None
    votos = Counter(modelo.predict(extract_feature_matrix(np.stack(ventanas))).tolist())
    return NOMBRES[votos.most_common(1)[0][0]]


def resumir(filas):
    meta = [f for f in filas if f["snr"] >= 10 and f["n_simbolos"] >= 512]
    tiempos = [f["tiempo_s"] for f in filas if f["n_simbolos"] >= 512]
    salida = {"todas": metricas(filas, NOMBRES), "snr_ge_10": metricas([f for f in filas if f["snr"] >= 10], NOMBRES),
              "meta_snr_ge_10_y_512_simbolos": metricas(meta, NOMBRES),
              "tiempo_s_mediana": float(np.median(tiempos)), "tiempo_s_p95": float(np.percentile(tiempos, 95)),
              "detectadas": float(np.mean([f["detectada"] is True or f["detectada"] == 1 for f in filas]))}
    for s in sorted({f["snr"] for f in filas}):
        salida[f"snr_{s}"] = metricas([f for f in filas if f["snr"] == s], NOMBRES)
    return salida


def confusion(filas):
    columnas = [*NOMBRES, DESCONOCIDA, "INDETERMINADA"]
    reales = list(dict.fromkeys(f["real"] for f in filas))
    return {"columnas": columnas,
            "filas": {r: [sum(f["real"] == r and f["prediccion"] == c for f in filas) for c in columnas] for r in reales}}


def extension_catalogo(cadena):
    paquete = cargar_paquete(RUTA_PAQUETE)
    puntos = g.desconocidas_prueba()["16APSK"]
    grupo = {"16APSK": (puntos, "16APSK")}
    items = conjunto_repo(2026111850, grupo, por_caso=6, snrs=(10, 15, 20, 30))
    antes = [analizar_cascada(it["iq"], paquete)["modulacion"] for it in items]
    z = inscribir_modulacion(puntos, cadena, 24, 2026111851)
    paquete["decisor"].inscribir("16APSK", z)
    despues = [analizar_cascada(it["iq"], paquete)["modulacion"] for it in items]
    return {"n": len(items), "antes": dict(Counter(antes)), "despues": dict(Counter(despues)),
            "reconocida_antes": float(np.mean([a == "16APSK" for a in antes])),
            "reconocida_despues": float(np.mean([a == "16APSK" for a in despues]))}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    if (SALIDA / "metricas.json").exists():
        raise SystemExit("La evaluación final ya se ejecutó; no se repite.")
    paquete = cargar_paquete(RUTA_PAQUETE)
    svm = entrenar_svm()
    resultado = {"paquete_sha256": hashlib.sha256(RUTA_PAQUETE.read_bytes()).hexdigest(),
                 "cadena": paquete["cadena"], "generadores": {}}
    for generador in ("gnuradio", "independiente"):
        items = cargar_items(generador)
        salidas = Parallel(n_jobs=12)(delayed(analizar)(it) for it in items)
        filas = [fila_final(it, r) for it, r in zip(items, salidas)]
        conocidos = [it for it in items if it["etiqueta"] in NOMBRES]
        predicciones = [svm_archivo(svm, it) for it in conocidos]
        aciertos = [p == it["etiqueta"] for p, it in zip(predicciones, conocidos)]
        altos = [a for a, it in zip(aciertos, conocidos) if it["snr"] >= 10]
        resultado["generadores"][generador] = {
            "cascada": resumir(filas), "confusion": confusion(filas),
            "svm_referencia": {"exactitud_cerrada": float(np.mean(aciertos)), "snr_ge_10": float(np.mean(altos)),
                               "sps_8_snr_ge_10": float(np.mean([a for a, it in zip(aciertos, conocidos)
                                                                 if it["snr"] >= 10 and it["sps"] == 8] or [np.nan]))},
            "tasa_correcta": float(np.mean([abs(f["sps_est"] - f["sps"]) / f["sps"] < 0.01 for f in filas if f["sps_est"]])),
        }
        (SALIDA / f"filas_{generador}.json").write_text(json.dumps(filas, default=float), encoding="utf-8")
    resultado["extension_catalogo_16APSK"] = extension_catalogo(paquete["cadena"])
    (SALIDA / "metricas.json").write_text(json.dumps(resultado, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: {"cascada_meta": v["cascada"]["meta_snr_ge_10_y_512_simbolos"], "svm": v["svm_referencia"],
                          "tasa": v["tasa_correcta"], "tiempo_s_p95": v["cascada"]["tiempo_s_p95"]}
                      for k, v in resultado["generadores"].items()}, indent=1, default=float))
    print(json.dumps(resultado["extension_catalogo_16APSK"], indent=1))


if __name__ == "__main__":
    main()
