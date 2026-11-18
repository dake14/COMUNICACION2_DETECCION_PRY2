"""H2: compara el recuperador con la cadena clásica en canales no usados al entrenar."""
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from experiments.comun_cascada import RUTA_RECUPERADOR, canales_config
from src.cascada import geometria as g
from src.cascada.autofoco import autofoco
from src.cascada.cadena import simbolos_de
from src.cascada.evaluacion import evm_referencia
from src.cascada.recuperador import cargar_recuperador
from src.cascada.sintesis import Condicion, canal_aleatorio, transmitir

_MODELO = None


def _modelo():
    global _MODELO
    if _MODELO is None:
        torch.set_num_threads(1)
        _MODELO = cargar_recuperador(RUTA_RECUPERADOR)
    return _MODELO


def canales_prueba(rng):
    c = {"ninguno": np.array([1 + 0j])}
    for i, t in enumerate(canales_config("shift_channels")):
        c[f"shift{i}"] = t
    c["denso32"] = canal_aleatorio(rng, retardo_max=32, trayectos_max=8)
    return c


def caso(semilla, nombre, puntos, nombre_canal, snr):
    rng = np.random.default_rng(semilla)
    taps = canales_prueba(rng)[nombre_canal]
    cond = Condicion(snr_db=snr, beta=float(rng.choice([0.25, 0.35, 0.5])), taps=taps, tau=rng.uniform(0, 8),
                     fase=rng.uniform(-3, 3), cfo=rng.uniform(-0.002, 0.002))
    r = transmitir(puntos, 1024, cond, rng)
    fila = {"nombre": nombre, "canal": nombre_canal, "snr": snr}
    for cadena in ("directa", "clasica", "red"):
        z, _, _ = autofoco(simbolos_de(r["iq"], cadena, _modelo() if cadena == "red" else None))
        e = evm_referencia(z[32:-32], r["simbolos"][32:-32])
        fila[cadena] = float(20 * np.log10(max(e, 1e-3)))
    return fila


def main():
    puntos = {**g.catalogo(), **g.desconocidas_validacion()}
    semillas = np.random.SeedSequence(2026111803).generate_state(4000).tolist()
    tareas = [(n, p, c, s) for n, p in puntos.items()
              for c in ("ninguno", "shift0", "shift1", "denso32") for s in (10, 20, 30) for _ in range(5)]
    filas = Parallel(n_jobs=8)(delayed(caso)(semillas[i], *t) for i, t in enumerate(tareas))
    resumen = {}
    for canal in ("ninguno", "shift0", "shift1", "denso32"):
        f = [x for x in filas if x["canal"] == canal]
        resumen[canal] = {c: round(float(np.median([x[c] for x in f])), 2) for c in ("directa", "clasica", "red")}
    con_canal = [x for x in filas if x["canal"] != "ninguno"]
    mejora = float(np.mean([x["clasica"] - x["red"] for x in con_canal]))
    mejora_directa = float(np.mean([min(x["clasica"], x["directa"]) - x["red"] for x in con_canal]))
    desconocidas = [x for x in filas if x["nombre"] in g.desconocidas_validacion() and x["canal"] == "ninguno"]
    resumen["mejora_db_red_vs_clasica_con_canal"] = round(mejora, 2)
    resumen["mejora_db_red_vs_mejor_clasica"] = round(mejora_directa, 2)
    resumen["desconocidas_sin_canal_red_menos_directa_db"] = round(float(np.mean([x["red"] - x["directa"] for x in desconocidas])), 2)
    resumen["por_modulacion_con_canal"] = {
        n: {c: round(float(np.median([x[c] for x in con_canal if x["nombre"] == n])), 2)
            for c in ("directa", "clasica", "red")} for n in puntos}
    resumen["cadena_elegida"] = "red" if mejora_directa >= 3.0 else "clasica"
    Path("results/cascada/h2.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    print(json.dumps(resumen, indent=1))


if __name__ == "__main__":
    main()
