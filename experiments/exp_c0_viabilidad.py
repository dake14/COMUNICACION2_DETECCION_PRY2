"""H0: viabilidad del normalizador y del receptor clásico, sin redes."""
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.cascada import geometria as g
from src.cascada.autofoco import autofoco
from src.cascada.clasica import simbolos_clasicos, simbolos_sin_ecualizar
from src.cascada.evaluacion import evm_ciego
from src.cascada.normalizador import normalizar
from src.cascada.sintesis import Condicion, canal_aleatorio, transmitir

SALIDA = Path("results/cascada")
SPS = [4.0, 5.3, 8.0, 10.0, 13.0]
SNR = [0, 5, 10, 20]
BETAS = [0.2, 0.35, 0.5]


def caso_tasa(semilla, nombre, sps, snr, beta):
    rng = np.random.default_rng(semilla)
    cond = Condicion(snr_db=snr, beta=beta, tau=rng.uniform(0, 8), fase=rng.uniform(-3, 3),
                     cfo=rng.uniform(-0.01, 0.01))
    if rng.uniform() < 0.5:
        cond.taps = canal_aleatorio(rng, retardo_max=8, trayectos_max=3)
    iq = transmitir(g.catalogo()[nombre], 512, cond, rng, sps_salida=sps)["iq"]
    hip = normalizar(iq)
    errores = [abs(h.sps_origen - sps) / sps for h in hip]
    return {"nombre": nombre, "sps": sps, "snr": snr, "beta": beta,
            "top1": bool(errores[0] < 0.01), "top3": bool(min(errores) < 0.01)}


def caso_evm(semilla, nombre, snr, con_canal):
    rng = np.random.default_rng(semilla)
    puntos = g.catalogo()[nombre]
    cond = Condicion(snr_db=snr, beta=0.35, tau=rng.uniform(0, 8), cfo=rng.uniform(-0.0015, 0.0015),
                     fase=rng.uniform(-3, 3))
    if con_canal:
        cond.taps = canal_aleatorio(rng, retardo_max=16, trayectos_max=3)
    iq = transmitir(puntos, 1024, cond, rng)["iq"]
    sin, _, _ = autofoco(simbolos_sin_ecualizar(iq))
    con, _, _ = autofoco(simbolos_clasicos(iq))
    return {"nombre": nombre, "snr": snr, "canal": con_canal,
            "evm_sin": evm_ciego(sin[64:], puntos), "evm_cma": evm_ciego(con[64:], puntos)}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    semillas = np.random.SeedSequence(2026111800).generate_state(4000).tolist()
    tareas = [(n, s, r, b) for n in g.catalogo() for s in SPS for r in SNR for b in BETAS]
    tareas = tareas * 2
    tasa = Parallel(n_jobs=6)(delayed(caso_tasa)(semillas[i], *t) for i, t in enumerate(tareas))
    t_evm = [(n, r, c) for n in g.catalogo() for r in (10, 20, 30) for c in (False, True)] * 8
    evm = Parallel(n_jobs=6)(delayed(caso_evm)(semillas[2000 + i], *t) for i, t in enumerate(t_evm))

    resumen = {"tasa_por_snr": {}, "tasa_por_sps": {}, "tasa_por_beta": {}, "evm": {}}
    for r in SNR:
        f = [x for x in tasa if x["snr"] == r]
        resumen["tasa_por_snr"][r] = {"top1": np.mean([x["top1"] for x in f]), "top3": np.mean([x["top3"] for x in f])}
    altos = [x for x in tasa if x["snr"] >= 5]
    for s in SPS:
        resumen["tasa_por_sps"][s] = float(np.mean([x["top1"] for x in altos if x["sps"] == s]))
    for b in BETAS:
        resumen["tasa_por_beta"][b] = float(np.mean([x["top1"] for x in altos if x["beta"] == b]))
    for c in (False, True):
        for r in (10, 20, 30):
            f = [x for x in evm if x["canal"] == c and x["snr"] == r]
            resumen["evm"][f"canal={c},snr={r}"] = {
                k: {n: round(float(np.median([x[k] for x in f if x["nombre"] == n])), 3) for n in g.catalogo()}
                for k in ("evm_sin", "evm_cma")}
    resumen["criterio_tasa_top1_snr_ge_5"] = float(np.mean([x["top1"] for x in altos]))
    resumen["criterio_tasa_top3_snr_ge_5"] = float(np.mean([x["top3"] for x in altos]))
    (SALIDA / "h0.json").write_text(json.dumps(resumen, indent=1, default=float), encoding="utf-8")
    print(json.dumps(resumen, indent=1, default=float))


if __name__ == "__main__":
    main()
