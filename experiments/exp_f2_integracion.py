"""Etapa 2b: integra el enrutador de familias con la cascada de constelación.

1. ``huellas``: para cada archivo, la huella de constelación de la cascada robusta (y, en el banco de evaluación, la
   salida del enrutador, la tasa ciega y los tonos). Caché en ``results/familias/integracion/huellas_<banco>.npz``.
2. ``inscribir``: inscribe en el decisor las 14 modulaciones de constelación con las huellas de
   ``digitales_entreno`` (sin tocar las redes) y recalibra temperatura, desplazamiento de la puerta y umbral con una
   parte apartada de ese banco. Guarda ``results/familias/integracion/paquete_integrado.joblib``.
3. ``evaluar``: compara enrutador solo, cascada sola y la fusión P(familia) × P(modulación | familia) en el banco
   ``digitales`` (mismos generadores, archivos distintos). Escribe ``results/familias/integracion/resumen.json``.

    .venv/Scripts/python.exe -m experiments.exp_f2_integracion --etapa todo
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import copy
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed

from src.banco.modulaciones import catalogo_digital
from src.cascada.cadena import recuperar
from src.cascada.decisor import DESCONOCIDA, INDETERMINADA
from src.cascada.detector import snr_ciega_db
from src.cascada.pipeline import calidad_de, cargar_paquete, huella_de_simbolos
from src.familias.enrutador import Enrutador
from src.familias.integracion import fusionar
from src.familias.tasa import estimar_tasa
from src.familias.tonos import analizar_fsk

# ─────────────────────────────  Config  ──────────────────────────────
BANCOS = {"entreno": Path("data/banco/digitales_entreno"), "evaluacion": Path("data/banco/digitales")}
PAQUETE_ROBUSTO = Path("results/cascada/modelos_robustos/paquete_adversaria_desplazada.joblib")
MODELO_ENRUTADOR = Path("results/familias/modelo")
SALIDA = Path("results/familias/integracion")
FAMILIA_FRECUENCIA = "frecuencia"
PAQUETE_INTEGRADO = SALIDA / "paquete_integrado.joblib"
FAMILIA_DE = {k: m.familia for k, m in catalogo_digital().items()}
CLASES_CONSTELACION = [k for k, f in FAMILIA_DE.items() if f != FAMILIA_FRECUENCIA]
DESPLAZAMIENTOS = np.round(np.arange(-2.0, 2.001, 0.1), 2)
MAX_CONOCIDAS_COMO_DESCONOCIDAS = 0.10
SNR_OPERACION = 10.0
MAX_CONOCIDAS_INDETERMINADAS = 0.10  # el umbral de 85 % de exactitud de la cascada dejaba 65 % sin decidir
UMBRALES = np.round(np.arange(0.2, 0.951, 0.01), 2)
PUNTAJE_TONOS = 0.8  # elegido con la partición de calibración: 1,7 % de FSK conocidas marcadas, 42,7 % de 3FSK
_C = {}


# ───────────────────────────────  Datos  ─────────────────────────────
def manifiesto(base):
    return [json.loads(x) for x in (base / "manifiesto.jsonl").read_text(encoding="utf-8").splitlines()]


def _modelos(con_enrutador):
    if "paquete" not in _C:
        torch.set_num_threads(1)
        _C["paquete"] = cargar_paquete(PAQUETE_ROBUSTO)
    if con_enrutador and "enrutador" not in _C:
        _C["enrutador"] = Enrutador.cargar(MODELO_ENRUTADOR)
    return _C["paquete"], _C.get("enrutador")


def _archivo(base, f, con_enrutador):
    x = np.fromfile(base / f["archivo"], np.complex64).astype(np.complex128)
    paquete, enrutador = _modelos(con_enrutador)
    salida = {"z": None, "calidad": None, "sps_cascada": np.nan}
    r = recuperar(x, paquete["cadena"], paquete.get("recuperador"))
    if r is not None:
        z, _ = huella_de_simbolos(r["simbolos"], paquete)
        salida.update(z=z[0], calidad=calidad_de(r, snr_ciega_db(x))[0], sps_cascada=float(r["sps_origen"]))
    if con_enrutador:
        with torch.no_grad():
            a = enrutador.analizar(x)
        tasa = estimar_tasa(x)
        tonos = analizar_fsk(x, tasa["sps"]) if tasa["sps"] else {"tonos": 0, "h": 0.0, "puntaje": 0.0}
        salida.update(p_enrutador=np.array(list(a["clases"].values())), sps_tasa=tasa["sps"] or np.nan,
                      tonos=tonos["tonos"], h=tonos["h"], puntaje_tonos=tonos["puntaje"])
    return salida


def huellas(nombre, trabajos=12):
    cache = SALIDA / f"huellas_{nombre}.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    base, con_enrutador = BANCOS[nombre], nombre == "evaluacion"
    filas = [f for f in manifiesto(base) if con_enrutador or f["familia"] != FAMILIA_FRECUENCIA]
    inicio = time.time()
    partes = Parallel(n_jobs=trabajos, batch_size=8)(delayed(_archivo)(base, f, con_enrutador) for f in filas)
    dim = next(len(p["z"]) for p in partes if p["z"] is not None)
    datos = {k: np.array([f[k] for f in filas]) for k in ("archivo", "modulacion", "etiqueta", "grupo", "familia",
                                                          "generador", "canal")}
    datos["snr"] = np.array([f["snr_db"] for f in filas], float)
    datos["ok"] = np.array([p["z"] is not None for p in partes])
    datos["z"] = np.stack([p["z"] if p["z"] is not None else np.full(dim, np.nan) for p in partes])
    datos["calidad"] = np.stack([p["calidad"] if p["z"] is not None else np.full(3, np.nan) for p in partes])
    datos["sps_cascada"] = np.array([p["sps_cascada"] for p in partes])
    if con_enrutador:
        for k in ("p_enrutador", "sps_tasa", "tonos", "h", "puntaje_tonos"):
            datos[k] = np.array([p[k] for p in partes])
        datos["clases_enrutador"] = np.array(json.loads((MODELO_ENRUTADOR / "info.json").read_text("utf-8"))["clases"])
    SALIDA.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **datos)
    print("huellas", nombre, len(filas), "ok", int(datos["ok"].sum()), f"{(time.time() - inicio) / 60:.1f} min",
          flush=True)
    return datos


# ─────────────────────────────  Inscribir  ───────────────────────────
def _particion(d):
    """Calibración: archivos con índice 0 del catálogo y desconocidas de validación. Evaluación: el resto."""
    indice = np.array([int(a.rsplit("_", 1)[1].split(".")[0]) for a in d["archivo"]])
    calibracion = ((d["grupo"] == "catalogo") & (indice == 0)) | (d["grupo"] == "validacion")
    return calibracion, ~calibracion


def _temperatura(decisor, z, etiquetas):
    l = decisor._verosimilitudes(z)
    y = np.array([decisor.nombres.index(e) for e in etiquetas])
    filas = np.arange(len(y))
    nll = lambda t: -np.mean(np.log(np.maximum(decisor._cerrado(l, t)[filas, y], 1e-12)))
    return float(min(np.geomspace(0.5, 100, 40), key=nll))


def inscribir_decisor(entreno, calib):
    datos = joblib.load(PAQUETE_ROBUSTO)
    decisor = copy.deepcopy(datos["decisor"])
    decisor.modelos = {}
    for clase in CLASES_CONSTELACION:
        decisor.inscribir(clase, entreno["z"][entreno["modulacion"] == clase])
    m = calib["ok"] & (calib["grupo"] == "catalogo") & np.isin(calib["modulacion"], CLASES_CONSTELACION)
    decisor.temperatura = _temperatura(decisor, calib["z"][m], calib["modulacion"][m])
    return datos, decisor


# ──────────────────────────────  Fusión  ─────────────────────────────
def _p_decisor(decisor, d):
    p = np.full((len(d["ok"]), len(decisor.nombres)), np.nan)
    ok = d["ok"]
    p[ok] = decisor.probabilidades(d["z"][ok], d["calidad"][ok])
    return p


def fusion_lote(decisor, d, p_dec, con_tonos=True):
    clases = list(d["clases_enrutador"])
    familias = sorted(set(FAMILIA_DE.values()))
    nombres = decisor.nombres
    salida = []
    for i in range(len(d["ok"])):
        pr = dict(zip(clases, d["p_enrutador"][i].tolist()))
        pf = {f: sum(v for c, v in pr.items() if FAMILIA_DE[c] == f) for f in familias}
        pd = dict(zip(nombres, p_dec[i].tolist())) if d["ok"][i] else None
        tonos = int(d["tonos"][i]) if con_tonos and d["puntaje_tonos"][i] >= PUNTAJE_TONOS else None
        salida.append(fusionar(pf, pr, pd, FAMILIA_DE, tonos)["probabilidades"])
    orden = [*clases, DESCONOCIDA]
    return orden, np.array([[s[k] for k in orden] for s in salida])


def _salidas(p, orden, umbral):
    k = p.argmax(1)
    return np.where(p[np.arange(len(p)), k] >= umbral, np.array(orden)[k], INDETERMINADA)


def _umbral(p, etiquetas):
    """El mayor umbral que deja como mucho un 10 % de conocidas sin decidir (mismo criterio que la puerta)."""
    confianza = p.max(1)[etiquetas != DESCONOCIDA]
    validos = [u for u in UMBRALES if np.mean(confianza < u) <= MAX_CONOCIDAS_INDETERMINADAS]
    return float(max(validos)) if validos else float(UMBRALES[0])


def calibrar_fusion(decisor, d, m, con_tonos):
    """Desplaza la puerta (como exp_c10, ahora sobre la salida fusionada) y elige el umbral."""
    regresion = decisor.puerta[-1]
    base = regresion.intercept_.copy()
    conocidas = m & (d["etiqueta"] != DESCONOCIDA) & (d["snr"] >= SNR_OPERACION)
    elegido = DESPLAZAMIENTOS[0]
    for desplazamiento in DESPLAZAMIENTOS:
        regresion.intercept_ = base + desplazamiento
        orden, p = fusion_lote(decisor, _sub(d, conocidas), _p_decisor(decisor, _sub(d, conocidas)), con_tonos)
        if np.mean(np.array(orden)[p.argmax(1)] == DESCONOCIDA) <= MAX_CONOCIDAS_COMO_DESCONOCIDAS:
            elegido = desplazamiento
    regresion.intercept_ = base + elegido
    alta = m & (d["snr"] >= SNR_OPERACION)
    orden, p = fusion_lote(decisor, _sub(d, alta), _p_decisor(decisor, _sub(d, alta)), con_tonos)
    decisor.umbral = _umbral(p, d["etiqueta"][alta])
    return float(elegido)


def _sub(d, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for k, v in d.items()}


def _rutas_desde_integracion(datos):
    """cargar_paquete resuelve las rutas contra la carpeta del paquete: se reescriben desde SALIDA."""
    relativa = lambda r: Path(os.path.relpath(PAQUETE_ROBUSTO.parent / r, SALIDA)).as_posix()
    meta = dict(datos["metadatos"])
    meta["descriptores"] = {k: relativa(v) for k, v in meta.get("descriptores", {}).items()}
    return {**datos, "ruta_recuperador": relativa(datos["ruta_recuperador"]) if datos.get("ruta_recuperador") else None,
            "ruta_descriptor": relativa(datos["ruta_descriptor"]), "metadatos": meta}


def inscribir(trabajos=12, con_tonos=True):
    entreno, evaluacion = huellas("entreno", trabajos), huellas("evaluacion", trabajos)
    calibracion, _ = _particion(evaluacion)
    datos, decisor = inscribir_decisor(entreno, _sub(evaluacion, calibracion))
    desplazamiento = calibrar_fusion(decisor, evaluacion, calibracion, con_tonos)
    print("temperatura", round(decisor.temperatura, 3), "desplazamiento", desplazamiento, "umbral",
          round(decisor.umbral, 3), flush=True)
    base = _rutas_desde_integracion(datos)
    paquete = {**base, "decisor": decisor,
               "enrutador": MODELO_ENRUTADOR.as_posix(), "familia_de": FAMILIA_DE, "con_tonos": con_tonos,
               "puntaje_tonos": PUNTAJE_TONOS,
               "metadatos": {**base["metadatos"], "integracion": {
                   "inscritas": CLASES_CONSTELACION, "desplazamiento_puerta": desplazamiento,
                   "calibracion": "digitales: catalogo con indice 0 + validacion"}}}
    joblib.dump(paquete, PAQUETE_INTEGRADO, compress=3)
    print("guardado", PAQUETE_INTEGRADO, flush=True)


# ─────────────────────────────  Evaluar  ─────────────────────────────
def _tasas(salida, etiqueta, mascara):
    conocida = mascara & (etiqueta != DESCONOCIDA)
    desconocida = mascara & (etiqueta == DESCONOCIDA)
    r = lambda m, v: round(float(np.mean(v[m])), 4) if m.any() else None
    return {"conocidas_correctas": r(conocida, salida == etiqueta),
            "conocidas_como_desconocidas": r(conocida, salida == DESCONOCIDA),
            "conocidas_indeterminadas": r(conocida, salida == INDETERMINADA),
            "desconocidas_detectadas": r(desconocida, salida == DESCONOCIDA),
            "desconocidas_no_aceptadas": r(desconocida, np.isin(salida, [DESCONOCIDA, INDETERMINADA])),
            "n_conocidas": int(conocida.sum()), "n_desconocidas": int(desconocida.sum())}


def _por(clave, salida, d, m):
    return {v: _tasas(salida, d["etiqueta"], m & (d[clave] == v))["conocidas_correctas"]
            for v in sorted(set(d[clave][m & (d["etiqueta"] != DESCONOCIDA)]))}


def evaluar(trabajos=12):
    d = huellas("evaluacion", trabajos)
    paquete = joblib.load(PAQUETE_INTEGRADO)
    decisor = paquete["decisor"]
    _, m = _particion(d)
    alta = m & (d["snr"] >= SNR_OPERACION)
    etiqueta = d["etiqueta"]
    orden, p = fusion_lote(decisor, d, _p_decisor(decisor, d), paquete["con_tonos"])
    fusionada = _salidas(p, orden, decisor.umbral)
    clases = list(d["clases_enrutador"])
    enrutador = np.array(clases)[d["p_enrutador"].argmax(1)]
    original = joblib.load(PAQUETE_ROBUSTO)["decisor"]
    p_orig = _p_decisor(original, d)
    cascada = np.where(d["ok"], _salidas(np.nan_to_num(p_orig), original.nombres, original.umbral), INDETERMINADA)
    seis = np.isin(d["modulacion"], [n for n in original.nombres if n != DESCONOCIDA]) | (etiqueta == DESCONOCIDA)
    argmax = np.array(orden)[p.argmax(1)]
    familia_pred = np.array([FAMILIA_DE.get(s, s) for s in argmax])
    familia_enrutador = np.array([FAMILIA_DE[s] for s in enrutador])
    conocida_alta = alta & (etiqueta != DESCONOCIDA)
    resumen = {
        "particion": "evaluación: catalogo con indice 1 + prueba (desconocidas nuevas)",
        "umbral": decisor.umbral, "temperatura": decisor.temperatura,
        "desplazamiento_puerta": paquete["metadatos"]["integracion"]["desplazamiento_puerta"],
        "con_tonos": paquete["con_tonos"],
        "fusion": {"todas": _tasas(fusionada, etiqueta, m), "snr>=10": _tasas(fusionada, etiqueta, alta)},
        "fusion_sin_umbral": {"snr>=10": _tasas(argmax, etiqueta, alta)},
        "enrutador_solo": {"todas": _tasas(enrutador, etiqueta, m), "snr>=10": _tasas(enrutador, etiqueta, alta)},
        "cascada_original_6_clases": {"snr>=10": _tasas(cascada, etiqueta, alta & seis)},
        "fusion_en_las_6_clases": {"snr>=10": _tasas(fusionada, etiqueta, alta & seis)},
        "familia_snr>=10": {"fusion": round(float(np.mean((familia_pred == d["familia"])[conocida_alta])), 4),
                            "enrutador": round(float(np.mean((familia_enrutador == d["familia"])[conocida_alta])), 4)},
        "por_clase_snr>=10": {"fusion": _por("modulacion", fusionada, d, alta),
                              "enrutador": _por("modulacion", enrutador, d, alta)},
        "por_familia_snr>=10": {"fusion": _por("familia", fusionada, d, alta),
                                "enrutador": _por("familia", enrutador, d, alta)},
        "por_canal_snr>=10": _por("canal", fusionada, d, alta),
        "desconocidas_snr>=10": {mod: {"detectada": round(float(np.mean(fusionada[x] == DESCONOCIDA)), 3),
                                       "no_aceptada": round(float(np.mean(np.isin(fusionada[x],
                                                                                  [DESCONOCIDA, INDETERMINADA]))), 3)}
                                 for mod in sorted(set(d["modulacion"][alta & (etiqueta == DESCONOCIDA)]))
                                 for x in [alta & (d["modulacion"] == mod)]},
    }
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in ("umbral", "desplazamiento_puerta", "fusion", "fusion_sin_umbral",
                                              "enrutador_solo", "cascada_original_6_clases", "fusion_en_las_6_clases",
                                              "familia_snr>=10", "desconocidas_snr>=10")}, indent=1),
          flush=True)
    return resumen


# ─────────────────────────────  Entry  ───────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", choices=("huellas", "inscribir", "evaluar", "todo"), default="todo")
    p.add_argument("--trabajos", type=int, default=12)
    p.add_argument("--sin-tonos", action="store_true", help="No usar el conteo de tonos para marcar desconocidas.")
    p.add_argument("--limite", type=int, default=None, help="Solo para medir el tiempo: N archivos, sin caché.")
    a = p.parse_args()
    if a.limite:
        base = BANCOS["evaluacion"]
        filas = manifiesto(base)[:: len(manifiesto(base)) // a.limite][: a.limite]
        inicio = time.time()
        Parallel(n_jobs=a.trabajos, batch_size=2)(delayed(_archivo)(base, f, True) for f in filas)
        print(f"{a.limite} archivos en {time.time() - inicio:.1f} s", flush=True)
        return
    if a.etapa in ("huellas", "todo"):
        for nombre in BANCOS:
            huellas(nombre, a.trabajos)
    if a.etapa in ("inscribir", "todo"):
        inscribir(a.trabajos, not a.sin_tonos)
    if a.etapa in ("evaluar", "todo"):
        evaluar(a.trabajos)


if __name__ == "__main__":
    main()
