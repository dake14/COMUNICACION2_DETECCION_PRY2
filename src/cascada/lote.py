"""Procesa una carpeta de archivos y produce la tabla del desafío."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

from src.cascada.decisor import DESCONOCIDA
from src.cascada.pipeline import PAQUETE_DEFECTO, analizar_cascada, cargar_paquete
from src.monitor import SUPPORTED_EXTENSIONS, load_signal


def agrupar_desconocidas(resultados, paquete):
    indices = [i for i, r in enumerate(resultados)
               if r["modulacion"] == DESCONOCIDA and r.get("huella") is not None]
    for r in resultados:
        r["grupo"] = None
    if len(indices) < 2:
        return resultados
    v = paquete["decisor"].proyectar(np.stack([resultados[i]["huella"] for i in indices]))
    etiquetas = HDBSCAN(min_cluster_size=2, min_samples=1, allow_single_cluster=True).fit_predict(v)
    for i, e in zip(indices, etiquetas):
        if e >= 0:
            resultados[i]["grupo"] = f"U{int(e) + 1}"
    return resultados


def procesar(carpeta, paquete, sps=None, dtype="complex64"):
    rutas = sorted(p for p in Path(carpeta).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    resultados = []
    for ruta in rutas:
        iq = load_signal(ruta, dtype=dtype)
        r = analizar_cascada(iq, paquete, sps=sps)
        r["senal"] = ruta.stem
        resultados.append(r)
    return agrupar_desconocidas(resultados, paquete)


def fila_tabla(r):
    detectada = r["deteccion"]["detectada"] if r.get("deteccion") else None
    modulacion = r["modulacion"] if detectada is not False else "NO_SIGNAL"
    snr = r.get("snr_ciega_db") if detectada else None
    descripcion = (r.get("descripcion") or {}) if detectada is not False else {}
    return {"Señal": r["senal"], "Señal detectada": {True: "Sí", False: "No", None: "Indeterminada"}[detectada],
            "SNR estimada (dB)": "" if snr is None else round(snr, 1), "Modulación estimada": modulacion,
            "Confianza": "" if r["confianza"] is None or detectada is False else round(r["confianza"], 3),
            "Grupo": r.get("grupo") or "",
            "Puntos aprox.": descripcion.get("puntos_aprox", ""),
            "Anillos aprox.": descripcion.get("anillos_aprox", "")}


def main():
    p = argparse.ArgumentParser(description="Clasifica una carpeta de archivos I/Q con la cascada.")
    p.add_argument("carpeta")
    p.add_argument("--salida", default="results/cascada/lote")
    p.add_argument("--paquete", default=str(PAQUETE_DEFECTO))
    p.add_argument("--sps", type=float, default=None)
    p.add_argument("--dtype", default="complex64")
    a = p.parse_args()
    resultados = procesar(a.carpeta, cargar_paquete(a.paquete), a.sps, a.dtype)
    destino = Path(a.salida)
    destino.mkdir(parents=True, exist_ok=True)
    filas = [fila_tabla(r) for r in resultados]
    if filas:
        with open(destino / "tabla.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
    limpios = [{k: v for k, v in r.items() if k != "huella"} for r in resultados]
    (destino / "detalle.json").write_text(json.dumps(limpios, indent=1, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    for fila in filas:
        print(fila)


if __name__ == "__main__":
    main()
