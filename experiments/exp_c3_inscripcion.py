"""Inscribe el catálogo, ajusta el fondo y calibra con el generador del repositorio."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from experiments.comun_cascada import (DIR_MODELOS, RUTA_DESCRIPTOR, RUTA_PAQUETE, RUTA_RECUPERADOR,
                                       conjunto_repo, fila_resultado, grupo_validacion, metricas)
from experiments.exp_c2_descriptor import VISTAS, grupo
from src.cascada import geometria as g
from src.cascada.cadena import recuperar
from src.cascada.decisor import Decisor
from src.cascada.descriptor import calcular_huellas, cargar_descriptor, imagen_densidad
from src.cascada.pipeline import analizar_cascada, cargar_paquete, guardar_paquete
from src.cascada.recuperador import cargar_recuperador

_CACHE = {}


def _piezas(cadena):
    if "d" not in _CACHE:
        torch.set_num_threads(1)
        _CACHE["d"] = cargar_descriptor(RUTA_DESCRIPTOR)
        _CACHE["r"] = cargar_recuperador(RUTA_RECUPERADOR) if cadena == "red" else None
    return _CACHE["d"], _CACHE["r"]


def huellas_de(imagenes):
    d = cargar_descriptor(RUTA_DESCRIPTOR)
    x = imagenes.reshape(-1, 2, 64, 64).astype(np.float32) / 255 * 1.5
    return calcular_huellas(d, x)[0]


def huella_item(item, cadena):
    d, r = _piezas(cadena)
    salida = recuperar(item["iq"], cadena, r)
    if salida is None:
        return None
    return calcular_huellas(d, imagen_densidad(salida["simbolos"])[None])[0][0]


def inscribir_modulacion(puntos, cadena, n_grupos, semilla):
    semillas = np.random.SeedSequence(semilla).generate_state(n_grupos).tolist()
    partes = Parallel(n_jobs=12)(delayed(grupo)(int(s), 12, cadena, puntos) for s in semillas)
    return huellas_de(np.concatenate([p[0] for p in partes]))


def analizar_item(item, ruta, cadena):
    if "p" not in _CACHE:
        torch.set_num_threads(1)
        _CACHE["p"] = cargar_paquete(ruta)
    return analizar_cascada(item["iq"], _CACHE["p"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cadena", required=True)
    a = p.parse_args()
    catalogo = {}
    for i, (nombre, puntos) in enumerate(g.catalogo().items()):
        catalogo[nombre] = inscribir_modulacion(puntos, a.cadena, 24, 2026111810 + i)
        print("inscrita", nombre, catalogo[nombre].shape, flush=True)
    semillas = np.random.SeedSequence(2026111820).generate_state(60).tolist()
    partes = Parallel(n_jobs=12)(delayed(grupo)(int(s), 20, a.cadena) for s in semillas)
    fondo = huellas_de(np.concatenate([p[0] for p in partes]))
    print("fondo", fondo.shape, flush=True)
    decisor = Decisor().ajustar(fondo, catalogo)

    grupo_val = grupo_validacion()
    items = conjunto_repo(2026111830, grupo_val, por_caso=10)
    z = Parallel(n_jobs=12)(delayed(huella_item)(it, a.cadena) for it in items)
    validos = [(it, v) for it, v in zip(items, z) if v is not None]
    z_val = np.stack([v for _, v in validos])
    decisor.calibrar(z_val, [it["etiqueta"] for it, _ in validos])
    print(json.dumps({"temperatura": decisor.temperatura, "previa_fondo": decisor.log_previa_fondo,
                      "umbral": decisor.umbral, "dim_pca": int(decisor.pca.n_components_)}), flush=True)
    guardar_paquete(RUTA_PAQUETE, decisor, a.cadena,
                    RUTA_RECUPERADOR.name if a.cadena == "red" else None, RUTA_DESCRIPTOR.name,
                    {"vistas_por_geometria": VISTAS})

    nombres = list(g.catalogo())
    items_ind = conjunto_repo(2026111831, grupo_val, por_caso=4)
    resultados = Parallel(n_jobs=12)(delayed(analizar_item)(it, RUTA_PAQUETE, a.cadena) for it in items_ind)
    filas = [fila_resultado(it, r, nombres) for it, r in zip(items_ind, resultados)]
    resumen = {"todas": metricas(filas, nombres)}
    for s in (0, 5, 10, 15, 20, 30):
        resumen[f"snr_{s}"] = metricas([f for f in filas if f["snr"] == s], nombres)
    resumen["snr_ge_10"] = metricas([f for f in filas if f["snr"] >= 10], nombres)
    Path("results/cascada/h3_validacion.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    print(json.dumps(resumen["snr_ge_10"], indent=1), flush=True)


if __name__ == "__main__":
    main()
