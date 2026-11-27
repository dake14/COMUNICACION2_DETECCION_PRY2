"""Especialistas dentro de cada familia de constelación: redes 1-D sobre histogramas finos de los símbolos recuperados.

1. ``simbolos``: guarda los símbolos que recupera la cascada robusta (hasta 2048 por archivo) de los dos bancos
   digitales. Caché en ``results/familias/especialistas_familia/simbolos_<banco>.npz``.
2. ``entrenar``: calcula ``vistas_constelacion`` (I, Q, radio y fase) y entrena un especialista por familia con
   ``digitales_entreno``. Guarda los modelos en ``results/familias/especialistas_familia/``.
3. ``medir``: exactitud dentro de la familia en la partición de calibración de ``digitales``, frente al enrutador, el
   decisor y sus productos.

    .venv/Scripts/python.exe -m experiments.exp_f3_especialistas_familia --etapa todo
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed

from experiments import exp_f2_integracion as f2
from experiments.exp_f2_integracion import BANCOS, FAMILIA_DE, FAMILIA_FRECUENCIA, PAQUETE_ROBUSTO, manifiesto
from src.cascada.cadena import recuperar
from src.cascada.pipeline import cargar_paquete
from src.familias.detalle import vistas_constelacion
from src.familias.especialistas import Especialista, entrenar, log_probabilidades

# ─────────────────────────────  Config  ──────────────────────────────
SALIDA = Path("results/familias/especialistas_familia")
MAX_SIMBOLOS = 2048
VENTANAS = 4
MIN_VENTANA = 256
PASOS = 4000
SEMILLA = 2026112701
GRUPOS = {"amplitud": ("amplitud",), "fase": ("fase",), "qam": ("qam",), "apsk": ("apsk",), "qam+apsk": ("qam", "apsk")}


def clases_de(grupo):
    return [c for c, f in FAMILIA_DE.items() if f in GRUPOS[grupo]]


_C = {}


# ───────────────────────────────  Datos  ─────────────────────────────
def _paquete():
    if "paquete" not in _C:
        torch.set_num_threads(1)
        _C["paquete"] = cargar_paquete(PAQUETE_ROBUSTO)
    return _C["paquete"]


def _simbolos(base, f):
    x = np.fromfile(base / f["archivo"], np.complex64).astype(np.complex128)
    p = _paquete()
    r = recuperar(x, p["cadena"], p.get("recuperador"))
    if r is None:
        return np.zeros(0, np.complex64)
    return np.asarray(r["simbolos"][:MAX_SIMBOLOS], np.complex64)


def simbolos(nombre, trabajos=12):
    cache = SALIDA / f"simbolos_{nombre}.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    base = BANCOS[nombre]
    filas = [f for f in manifiesto(base) if nombre != "entreno" or f["familia"] != FAMILIA_FRECUENCIA]
    inicio = time.time()
    partes = Parallel(n_jobs=trabajos, batch_size=8)(delayed(_simbolos)(base, f) for f in filas)
    largo = np.array([len(s) for s in partes])
    matriz = np.zeros((len(partes), MAX_SIMBOLOS), np.complex64)
    for i, s in enumerate(partes):
        matriz[i, :len(s)] = s
    SALIDA.mkdir(parents=True, exist_ok=True)
    np.savez(cache, archivo=np.array([f["archivo"] for f in filas]), simbolos=matriz, largo=largo)
    print("símbolos", nombre, len(filas), "sin recuperar", int((largo == 0).sum()),
          f"{(time.time() - inicio) / 60:.1f} min", flush=True)
    return {"archivo": np.array([f["archivo"] for f in filas]), "simbolos": matriz, "largo": largo}


# ─────────────────────────────  Vistas  ──────────────────────────────
def _ventanas(s, largo, rng, n):
    if largo < MIN_VENTANA or n == 1:
        return [vistas_constelacion(s[:largo])]
    salida = []
    for _ in range(n):
        tam = int(rng.integers(MIN_VENTANA, largo + 1))
        inicio = int(rng.integers(0, largo - tam + 1))
        salida.append(vistas_constelacion(s[inicio:inicio + tam]))
    return salida


def vistas_banco(nombre, ventanas, trabajos=12):
    cache = SALIDA / f"vistas_{nombre}_{ventanas}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["indice"]
    s = simbolos(nombre, trabajos)
    partes = Parallel(n_jobs=trabajos, batch_size=64)(
        delayed(_ventanas)(s["simbolos"][i], int(s["largo"][i]), np.random.default_rng([SEMILLA, i]), ventanas)
        for i in range(len(s["largo"])))
    indice = np.concatenate([np.full(len(p), i) for i, p in enumerate(partes)])
    x = np.stack([v for p in partes for v in p]).astype(np.float16)
    np.savez(cache, x=x, indice=indice)
    return x, indice


# ─────────────────────────────  Entrenar  ────────────────────────────
def _modulaciones(nombre):
    fila = {f["archivo"]: f["modulacion"] for f in manifiesto(BANCOS[nombre])}
    return np.array([fila[a] for a in simbolos(nombre)["archivo"]])


def entrenar_grupos(disp):
    x, indice = vistas_banco("entreno", VENTANAS)
    modulacion = _modulaciones("entreno")[indice]
    info = {}
    for k, grupo in enumerate(GRUPOS):
        clases = clases_de(grupo)
        m = np.isin(modulacion, clases)
        y = np.array([clases.index(c) for c in modulacion[m]])
        inicio = time.time()
        modelo, validacion = entrenar(x[m], y, len(clases), disp, PASOS, semilla=SEMILLA + k)
        torch.save({"config": modelo.config, "estado": modelo.cpu().state_dict(), "clases": clases},
                   SALIDA / f"especialista_{grupo}.pt")
        info[grupo] = {"clases": clases, "validacion_interna": round(validacion, 4), "muestras": int(m.sum())}
        print(grupo, info[grupo], f"{time.time() - inicio:.0f} s", flush=True)
    (SALIDA / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")


def cargar_especialista(grupo, disp="cpu"):
    d = torch.load(SALIDA / f"especialista_{grupo}.pt", map_location=disp, weights_only=True)
    modelo = Especialista(**d["config"])
    modelo.load_state_dict(d["estado"])
    return modelo.to(disp).eval(), d["clases"]


# ──────────────────────────────  Medir  ──────────────────────────────
def medir():
    d = f2.huellas("evaluacion")
    calibracion, _ = f2._particion(d)
    x, indice = vistas_banco("evaluacion", 1)
    assert (simbolos("evaluacion")["archivo"] == d["archivo"]).all()
    decisor = joblib.load(f2.PAQUETE_INTEGRADO)["decisor"]
    nombres = list(d["clases_enrutador"])
    resultados = {}
    for grupo in GRUPOS:
        modelo, clases = cargar_especialista(grupo)
        for alta in (False, True):
            snr_ok = d["snr"] >= f2.SNR_OPERACION if alta else np.ones(len(d["snr"]), bool)
            m = calibracion & np.isin(d["modulacion"], clases) & d["ok"] & snr_ok
            y = np.array([clases.index(c) for c in d["modulacion"][m]])
            pr = d["p_enrutador"][m][:, [nombres.index(c) for c in clases]]
            pd = decisor._cerrado(decisor._verosimilitudes(d["z"][m]), decisor.temperatura)[
                :, [decisor.nombres.index(c) for c in clases]]
            ps = np.exp(log_probabilidades(modelo, x[m]))
            exactitud = lambda p: round(float(np.mean(p.argmax(1) == y)), 4)
            resultados[f"{grupo}{' snr>=10' if alta else ''}"] = {
                "n": int(m.sum()), "enrutador": exactitud(pr), "decisor": exactitud(pd), "especialista": exactitud(ps),
                "enr×dec": exactitud(pr * pd), "enr×esp": exactitud(pr * ps), "enr×dec×esp": exactitud(pr * pd * ps)}
            print(grupo, "snr>=10" if alta else "todas", resultados[f"{grupo}{' snr>=10' if alta else ''}"], flush=True)
    (SALIDA / "medicion.json").write_text(json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────  Entry  ───────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", choices=("simbolos", "entrenar", "medir", "todo"), default="todo")
    p.add_argument("--trabajos", type=int, default=12)
    a = p.parse_args()
    if a.etapa in ("simbolos", "todo"):
        for nombre in BANCOS:
            simbolos(nombre, a.trabajos)
    if a.etapa in ("entrenar", "todo"):
        entrenar_grupos("cuda" if torch.cuda.is_available() else "cpu")
    if a.etapa in ("medir", "todo"):
        medir()


if __name__ == "__main__":
    main()
