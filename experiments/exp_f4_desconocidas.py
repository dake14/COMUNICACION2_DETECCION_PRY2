"""Etapa 5: desconocidas parecidas a una conocida (12PSK, 128QAM) con la prueba de ajuste a la constelación.

1. ``medir``: para cada archivo de los dos bancos digitales, el ajuste de sus símbolos recuperados a la constelación
   ideal de cada clase conocida (``src/familias/ajuste.py``). Caché en
   ``results/familias/desconocidas/mediciones_<banco>.npz``.
2. ``seleccionar``: los umbrales de cada clase salen de ``digitales_entreno`` (unos 400 archivos por clase a SNR >= 10;
   con los ~100 de la partición de calibración el cuantil del 2 % era ruidoso y OOK fallaba el 13 % en evaluación).
   Con la partición de calibración elige ``beta`` (cuán estricto es el umbral de cada clase) y el peso que conserva
   una clase inconsistente. La puerta y el umbral se recalibran en cada combinación. Escribe
   ``results/familias/desconocidas/seleccion.json``; los valores elegidos se copian a ``exp_f2_integracion``.

    .venv/Scripts/python.exe -m experiments.exp_f4_desconocidas --etapa todo
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from experiments import exp_f2_integracion as f2
from src.banco.modulaciones import catalogo_digital
from src.cascada import geometria as g
from src.cascada.decisor import DESCONOCIDA
from src.familias.ajuste import Consistencia, medir

# ─────────────────────────────  Config  ──────────────────────────────
SALIDA = Path("results/familias/desconocidas")
SIMBOLOS = "results/familias/especialistas_familia/simbolos_{}.npz"
BETAS = (0.02, 0.05, 0.10)
PESOS = (0.0, 0.2, 0.5)


def plantillas():
    """Constelación ideal de cada clase de constelación, como la ve el receptor después de recuperar los símbolos."""
    salida = {k: np.asarray(m.puntos, np.complex128) for k, m in catalogo_digital().items() if m.puntos}
    salida["PI4DQPSK"] = g.psk(8)
    return salida


# ─────────────────────────────  Medir  ───────────────────────────────
def mediciones(banco="evaluacion", trabajos=12):
    cache = SALIDA / f"mediciones_{banco}.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    simbolos = np.load(SIMBOLOS.format(banco))
    largo, matriz, archivo = simbolos["largo"], simbolos["simbolos"], simbolos["archivo"]
    plant = plantillas()
    inicio = time.time()
    filas = Parallel(n_jobs=trabajos, batch_size=8)(
        delayed(medir)(matriz[i, :largo[i]], plant) for i in range(len(largo)))
    valido = np.array([f is not None for f in filas])
    clases = list(plant)
    error = np.full((len(filas), len(clases)), np.nan)
    ocupacion = np.full_like(error, np.nan)
    libre = np.full(len(filas), np.nan)
    for i, f in enumerate(filas):
        if f is not None:
            libre[i] = f["libre"]
            error[i] = [f["plantillas"][c]["error"] for c in clases]
            ocupacion[i] = [f["plantillas"][c]["ocupacion"] for c in clases]
    SALIDA.mkdir(parents=True, exist_ok=True)
    salida = {"archivo": archivo, "valido": valido, "libre": libre, "error": error, "ocupacion": ocupacion,
              "clases": np.array(clases)}
    np.savez(cache, **salida)
    print("mediciones", banco, len(filas), "válidas", int(valido.sum()), f"{(time.time() - inicio) / 60:.1f} min", flush=True)
    return salida


def como_dicts(m):
    clases = list(m["clases"])
    return [{"libre": float(m["libre"][i]),
             "plantillas": {c: {"error": float(m["error"][i, k]), "ocupacion": float(m["ocupacion"][i, k])}
                            for k, c in enumerate(clases)}} if m["valido"][i] else None
            for i in range(len(m["valido"]))]


def consistencia_entreno(beta, trabajos=12):
    """Umbrales por clase con los archivos de ``digitales_entreno`` a SNR >= 10 dB."""
    m = mediciones("entreno", trabajos)
    fila = {f["archivo"]: f for f in f2.manifiesto(f2.BANCOS["entreno"])}
    modulacion = np.array([fila[a]["modulacion"] for a in m["archivo"]])
    snr = np.array([fila[a]["snr_db"] for a in m["archivo"]])
    lista = como_dicts(m)
    i = np.flatnonzero(m["valido"] & (snr >= f2.SNR_OPERACION))
    return Consistencia(beta).ajustar([lista[k] for k in i], list(modulacion[i]))


def con_ajuste(d, lista, consistencia, peso):
    pesos = np.empty(len(lista), object)
    for i, medicion in enumerate(lista):
        pesos[i] = {c: (1.0 if ok else peso) for c, ok in consistencia.consistentes(medicion).items()}
    return {**d, "consistentes": pesos}


# ────────────────────────────  Seleccionar  ──────────────────────────
def _metricas(decisor, d, m):
    orden, p = f2.fusion_lote(decisor, f2._sub(d, m), f2._p_decisor(decisor, f2._sub(d, m)))
    salida = f2._salidas(p, orden, decisor.umbral)
    etiqueta, mod = d["etiqueta"][m], d["modulacion"][m]
    t = f2._tasas(salida, etiqueta, np.ones(len(salida), bool))
    t["por_desconocida"] = {x: round(float(np.mean(salida[mod == x] == DESCONOCIDA)), 3)
                            for x in sorted(set(mod[etiqueta == DESCONOCIDA]))}
    return t


def elegir(filas):
    """La de mayor puntaje entre las que marcan como mucho un 10 % de conocidas como desconocidas (como la puerta)."""
    validas = [f for f in filas if f["beta"] is not None
               and f["conocidas_como_desconocidas"] <= f2.MAX_CONOCIDAS_COMO_DESCONOCIDAS]
    return max(validas, key=lambda f: f["puntaje"])


def seleccionar(trabajos=12):
    entreno, ev = f2.huellas("entreno", trabajos), f2.con_especialista(f2.huellas("evaluacion", trabajos))
    m = mediciones("evaluacion", trabajos)
    assert (m["archivo"] == ev["archivo"]).all()
    lista = como_dicts(m)
    calibracion, _ = f2._particion(ev)
    _, base = f2.inscribir_decisor(entreno, f2._sub(ev, calibracion))
    alta = calibracion & (ev["snr"] >= f2.SNR_OPERACION)
    filas = []
    for beta, peso in [(None, None)] + [(b, w) for b in BETAS for w in PESOS]:
        decisor = copy.deepcopy(base)
        d = ev if beta is None else con_ajuste(ev, lista, consistencia_entreno(beta, trabajos), peso)
        desplazamiento = f2.calibrar_fusion(decisor, d, calibracion, True)
        t = _metricas(decisor, d, alta)
        fila = {"beta": beta, "peso": peso, "desplazamiento": desplazamiento, "umbral": decisor.umbral, **t,
                "puntaje": round((t["conocidas_correctas"] + t["desconocidas_detectadas"]) / 2, 4)}
        filas.append(fila)
        print(json.dumps(fila, ensure_ascii=False), flush=True)
    elegida = elegir(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "seleccion.json").write_text(json.dumps({"elegida": elegida, "filas": filas}, indent=1,
                                                      ensure_ascii=False), encoding="utf-8")
    print("elegida", elegida["beta"], elegida["peso"], elegida["puntaje"], "sin ajuste", filas[0]["puntaje"],
          flush=True)


# ─────────────────────────────  Entry  ───────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", choices=("medir", "seleccionar", "todo"), default="todo")
    p.add_argument("--trabajos", type=int, default=12)
    a = p.parse_args()
    if a.etapa in ("medir", "todo"):
        for banco in ("evaluacion", "entreno"):
            mediciones(banco, a.trabajos)
    if a.etapa in ("seleccionar", "todo"):
        seleccionar(a.trabajos)


if __name__ == "__main__":
    main()
