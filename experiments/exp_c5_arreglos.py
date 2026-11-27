"""Compara arreglos de la cascada con señales nuevas: acierto frente a costo.

Cada arreglo inscribe su catálogo y calibra su decisor con las mismas semillas de H3.
El descriptor es el mismo para todos. Regla fijada antes de ver resultados: el arreglo más
eficiente es el más rápido a 2 puntos o menos del mejor en exactitud y en puntaje abierto.
"""
import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import binomtest

from experiments.comun_cascada import (RUTA_DESCRIPTOR, RUTA_RECUPERADOR, conjunto_procedural, conjunto_repo,
                                       fila_resultado, grupo_validacion, metricas)
from experiments.exp_c2_descriptor import VISTAS
from experiments.exp_c3_inscripcion import huellas_de
from experiments.exp_c4_final import entrenar_svm, svm_archivo
from src.cascada import geometria as g
from src.cascada.autofoco import entropia
from src.cascada.cadena import MAX_ENTROPIA, MIN_SIMBOLOS, _enfocar_ruta
from src.cascada.decisor import DESCONOCIDA, Decisor
from src.cascada.descriptor import calcular_huellas, cargar_descriptor, imagen_densidad
from src.cascada.detector import detectar_ciego, snr_ciega_db
from src.cascada.normalizador import normalizar
from src.cascada.pipeline import _indeterminada, calidad_de
from src.cascada.recuperador import cargar_recuperador
from src.cascada.sintesis import condicion_aleatoria, transmitir

DATOS = Path("data/cascada_arreglos")
SALIDA = Path("results/cascada/arreglos")
GENERADORES = ("gnuradio", "independiente")
NOMBRES = list(g.catalogo())
TRABAJOS = 8
TOLERANCIA = 0.02
GPU = "cuda"
REFERENCIA = "mixta_refino"
ARREGLOS = {
    "directa": (("directa",), False),
    "directa_refino": (("directa",), True),
    "cma": (("clasica",), False),
    "cma_refino": (("clasica",), True),
    "red": (("red",), False),
    "red_refino": (("red",), True),
    "clasicas_refino": (("clasica", "directa"), True),
    "mixta": (("red", "clasica", "directa"), False),
    "mixta_refino": (("red", "clasica", "directa"), True),
}
_C = {}


def modelos():
    if "d" not in _C:
        torch.set_num_threads(1)
        _C["d"] = cargar_descriptor(RUTA_DESCRIPTOR)
        _C["r"] = cargar_recuperador(RUTA_RECUPERADOR)
    return _C["d"], _C["r"]


def recuperar_arreglo(iq, nombre, sps=None):
    rutas, refinar = ARREGLOS[nombre]
    _, recuperador = modelos()
    for h in sorted(normalizar(iq, sps=sps), key=lambda h: -h.puntaje):
        if len(h.iq) < MIN_SIMBOLOS * 8:
            continue
        mejor = None
        for ruta in rutas:
            salida = _enfocar_ruta(h.iq, ruta, recuperador, refinar)
            if salida is None:
                continue
            valor = entropia(salida[0][:MAX_ENTROPIA])
            if mejor is None or valor < mejor[0]:
                mejor = (valor, *salida, ruta)
        if mejor is not None:
            _, z, df, nitidez, ruta = mejor
            return {"simbolos": z, "df": df, "nitidez": nitidez, "sps_origen": h.sps_origen, "ruta": ruta}
    return None


def vista(puntos, rng, nombre):
    n = int(rng.choice([128, 256, 512, 1024, 2048], p=[0.1, 0.2, 0.3, 0.25, 0.15]))
    sps = float(rng.choice([8.0, 4.0, 5.3, 6.0, 10.0, 12.0], p=[0.6, 0.1, 0.075, 0.075, 0.075, 0.075]))
    cond = condicion_aleatoria(rng, snr_rango=(0.0, 30.0), cfo_max=0.008, prob_canal=0.6)
    iq = transmitir(puntos, n, cond, rng, sps_salida=sps)["iq"]
    r = recuperar_arreglo(iq, nombre, sps=sps)
    return None if r is None else (np.clip(imagen_densidad(r["simbolos"]) / 1.5, 0, 1) * 255).astype(np.uint8)


def grupo(semilla, n_geometrias, nombre, puntos_fijos=None):
    rng = np.random.default_rng(semilla)
    imagenes = []
    for _ in range(n_geometrias):
        puntos = puntos_fijos if puntos_fijos is not None else g.geometria_aleatoria(rng)
        vistas = [vista(puntos, rng, nombre) for _ in range(VISTAS)]
        if all(v is not None for v in vistas):
            imagenes.append(np.stack(vistas))
    return np.stack(imagenes)


def huellas_grupos(semilla, n_grupos, n_geometrias, nombre, puntos=None):
    semillas = np.random.SeedSequence(semilla).generate_state(n_grupos).tolist()
    partes = Parallel(n_jobs=TRABAJOS)(delayed(grupo)(int(s), n_geometrias, nombre, puntos) for s in semillas)
    return huellas_de(np.concatenate(partes))


def huella_item(item, nombre):
    d, _ = modelos()
    x = np.asarray(item["iq"], dtype=np.complex128)
    r = recuperar_arreglo(x, nombre)
    if r is None:
        return None
    return calcular_huellas(d, imagen_densidad(r["simbolos"])[None])[0][0], calidad_de(r, snr_ciega_db(x))[0]


def ajustar(nombre, escala):
    torch.set_num_threads(8)
    catalogo = {m: huellas_grupos(2026111810 + i, max(2, 24 // escala), 12, nombre, p)
                for i, (m, p) in enumerate(g.catalogo().items())}
    fondo = huellas_grupos(2026111820, max(4, 60 // escala), 20, nombre)
    decisor = Decisor().ajustar(fondo, catalogo)
    grupos = grupo_validacion()
    items = (conjunto_repo(2026111830, grupos, por_caso=max(1, 10 // escala))
             + conjunto_repo(2026111831, grupos, por_caso=max(1, 4 // escala))
             + conjunto_procedural(2026111833, 200 // escala, 1200 // escala))
    salidas = Parallel(n_jobs=TRABAJOS)(delayed(huella_item)(it, nombre) for it in items)
    validos = [(it, v) for it, v in zip(items, salidas) if v is not None]
    return decisor.calibrar(np.stack([v[0] for _, v in validos]), [it["etiqueta"] for it, _ in validos],
                            np.stack([v[1] for _, v in validos]))


def analizar(iq, nombre, decisor):
    d, _ = modelos()
    x = np.asarray(iq, dtype=np.complex128)
    detectar_ciego(x)
    snr = snr_ciega_db(x)
    r = recuperar_arreglo(x, nombre)
    if r is None:
        return _indeterminada("tasa_no_identificada"), None
    z = calcular_huellas(d, imagen_densidad(r["simbolos"])[None])[0]
    return {**decisor.decidir(z, calidad_de(r, snr)), "etapas": {"sps_origen": r["sps_origen"]}}, r["ruta"]


def fila_archivo(item, nombre, ruta_decisor):
    clave = ("decisor", str(ruta_decisor))
    if clave not in _C:
        _C[clave] = joblib.load(ruta_decisor)
    r, ruta = analizar(item["iq"], nombre, _C[clave])
    return {**fila_resultado(item, r, NOMBRES), "generador": item["generador"], "archivo": item["archivo"],
            "n_simbolos": item["n_simbolos"], "ruta": ruta}


def cargar_items(limite=None):
    items = []
    for generador in GENERADORES:
        base = DATOS / generador
        manifiesto = json.loads((base / "manifiesto.json").read_text(encoding="utf-8"))
        for m in manifiesto[::limite or 1]:
            items.append({**m, "generador": generador, "iq": np.fromfile(base / m["archivo"], np.complex64)})
    return items


def muestra_tiempo(items, n=40):
    salida = []
    for generador in GENERADORES:
        grupo_g = [it for it in items if it["generador"] == generador and it["snr"] >= 10]
        salida += [grupo_g[int(i)] for i in np.linspace(0, len(grupo_g) - 1, n)]
    return salida


def medir(funcion, muestra, dispositivo="cpu"):
    """Mediana y percentil 95 en ms, archivo por archivo y con un hilo de CPU."""
    descriptor, recuperador = modelos()
    descriptor.to(dispositivo)
    recuperador.to(dispositivo)
    torch.set_num_threads(1)
    funcion(muestra[0])
    tiempos = []
    for it in muestra:
        inicio = time.perf_counter()
        funcion(it)
        tiempos.append(time.perf_counter() - inicio)
    descriptor.to("cpu")
    recuperador.to("cpu")
    return 1000 * float(np.median(tiempos)), 1000 * float(np.percentile(tiempos, 95))


def resumen_filas(filas):
    m = metricas(filas, NOMBRES)
    conocidas = [f for f in filas if f["etiqueta"] in NOMBRES]
    afirmadas = [f for f in conocidas + [f for f in filas if f["etiqueta"] == DESCONOCIDA] if f["prediccion"] in NOMBRES]
    con_tasa = [f for f in filas if f["sps_est"]]
    return {"exactitud": m["exactitud_cerrada"], "correctas_salida_real": m["exactitud_abierta"],
            "desconocidas_marcadas": m["rechazo_desconocidas"], "falsa_aceptacion": m["falsa_aceptacion"],
            "auroc": m["auroc_desconocidas"], "ece": m.get("ece"),
            "puntaje_abierto": (m["exactitud_abierta"] + m["rechazo_desconocidas"]) / 2,
            "precision_afirmadas": float(np.mean([f["prediccion"] == f["etiqueta"] for f in afirmadas])) if afirmadas else None,
            "tasa_correcta": float(np.mean([abs(f["sps_est"] - f["sps"]) / f["sps"] < 0.01 for f in con_tasa])),
            "n_conocidas": len(conocidas)}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - r, c + r]


def mcnemar(a, b):
    solo_a = sum(x and not y for x, y in zip(a, b))
    solo_b = sum(y and not x for x, y in zip(a, b))
    p = binomtest(solo_a, solo_a + solo_b, 0.5).pvalue if solo_a + solo_b else 1.0
    return {"solo_este": solo_a, "solo_referencia": solo_b, "p": float(p)}


def evaluar_arreglo(nombre, items, muestra, destino, escala):
    salida = destino / f"{nombre}.json"
    if salida.exists():
        return json.loads(salida.read_text(encoding="utf-8"))
    inicio = time.time()
    decisor = ajustar(nombre, escala)
    ruta = destino / "decisores" / f"{nombre}.joblib"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(decisor, ruta, compress=3)
    minutos_ajuste = (time.time() - inicio) / 60
    filas = Parallel(n_jobs=TRABAJOS)(delayed(fila_archivo)(it, nombre, ruta) for it in items)
    funcion = lambda it: analizar(it["iq"], nombre, decisor)
    cpu = medir(funcion, muestra)
    gpu = medir(funcion, muestra, GPU)
    datos = {"arreglo": nombre, "rutas": ARREGLOS[nombre][0], "refino": ARREGLOS[nombre][1],
             "usa_red": "red" in ARREGLOS[nombre][0], "minutos_ajuste": minutos_ajuste,
             "tiempo_ms_mediana": cpu[0], "tiempo_ms_p95": cpu[1],
             "tiempo_gpu_ms_mediana": gpu[0], "tiempo_gpu_ms_p95": gpu[1], "filas": filas}
    salida.write_text(json.dumps(datos, default=float), encoding="utf-8")
    print(nombre, f"listo en {(time.time() - inicio) / 60:.1f} min", flush=True)
    return datos


def evaluar_svm(items, muestra, destino):
    salida = destino / "svm.json"
    if salida.exists():
        return json.loads(salida.read_text(encoding="utf-8"))
    modelo = entrenar_svm()
    filas = [{"generador": it["generador"], "archivo": it["archivo"], "etiqueta": it["etiqueta"], "snr": it["snr"],
              "cerrada": svm_archivo(modelo, it)} for it in items if it["etiqueta"] in NOMBRES]
    cpu = medir(lambda it: svm_archivo(modelo, it), muestra)
    datos = {"arreglo": "svm", "tiempo_ms_mediana": cpu[0], "tiempo_ms_p95": cpu[1], "filas": filas}
    salida.write_text(json.dumps(datos, default=float), encoding="utf-8")
    return datos


def aciertos(datos):
    filas = [f for f in datos["filas"] if f["snr"] >= 10 and f["etiqueta"] in NOMBRES]
    filas.sort(key=lambda f: (f["generador"], f["archivo"]))
    return [f["cerrada"] == f["etiqueta"] for f in filas]


def tabla(resultados, svm):
    base = aciertos(resultados[REFERENCIA])
    filas = []
    for nombre, datos in resultados.items():
        altas = [f for f in datos["filas"] if f["snr"] >= 10 and f["n_simbolos"] >= 512]
        total = resumen_filas(altas)
        a = aciertos(datos)
        tiempo = datos["tiempo_ms_mediana"]
        rutas = {r: float(np.mean([f["ruta"] == r for f in altas])) for r in datos["rutas"]}
        filas.append({"arreglo": nombre, "usa_red": datos["usa_red"], "refino": datos["refino"], **total,
                      "ic95_exactitud": wilson(total["exactitud"], len(a)),
                      "por_generador": {x: resumen_filas([f for f in altas if f["generador"] == x]) for x in GENERADORES},
                      "tiempo_ms_mediana": tiempo, "tiempo_ms_p95": datos["tiempo_ms_p95"],
                      "tiempo_gpu_ms_mediana": datos["tiempo_gpu_ms_mediana"],
                      "tiempo_gpu_ms_p95": datos["tiempo_gpu_ms_p95"],
                      "aciertos_por_segundo": total["exactitud"] / (tiempo / 1000), "uso_rutas": rutas,
                      "minutos_ajuste": datos["minutos_ajuste"],
                      "mcnemar_vs_actual": None if nombre == REFERENCIA else mcnemar(a, base)})
    mejor_e = max(f["exactitud"] for f in filas)
    mejor_p = max(f["puntaje_abierto"] for f in filas)
    aptos = [f for f in filas if f["exactitud"] >= mejor_e - TOLERANCIA and f["puntaje_abierto"] >= mejor_p - TOLERANCIA]
    eficiente = min(aptos, key=lambda f: f["tiempo_ms_mediana"])["arreglo"]
    eficiente_gpu = min(aptos, key=lambda f: f["tiempo_gpu_ms_mediana"])["arreglo"]
    for f in filas:
        f["pareto"] = not any(o is not f and o["tiempo_ms_mediana"] <= f["tiempo_ms_mediana"]
                              and o["exactitud"] >= f["exactitud"] and o["puntaje_abierto"] >= f["puntaje_abierto"]
                              for o in filas)
    filas.sort(key=lambda f: f["tiempo_ms_mediana"])
    a_svm = aciertos(svm)
    p_svm = float(np.mean(a_svm))
    por_generador = {x: float(np.mean([f["cerrada"] == f["etiqueta"] for f in svm["filas"]
                                       if f["generador"] == x and f["snr"] >= 10])) for x in GENERADORES}
    return {"regla": f"el más rápido a {TOLERANCIA:.2f} o menos del mejor en exactitud y puntaje abierto",
            "mas_eficiente": eficiente, "mas_eficiente_gpu": eficiente_gpu, "mejor_exactitud": mejor_e, "mejor_puntaje_abierto": mejor_p,
            "n_conocidas": len(base), "arreglos": filas,
            "svm": {"exactitud": p_svm, "ic95_exactitud": wilson(p_svm, len(a_svm)), "por_generador": por_generador,
                    "tiempo_ms_mediana": svm["tiempo_ms_mediana"], "tiempo_ms_p95": svm["tiempo_ms_p95"],
                    "aciertos_por_segundo": p_svm / (svm["tiempo_ms_mediana"] / 1000),
                    "mcnemar_vs_actual": mcnemar(a_svm, base)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rapido", action="store_true", help="prueba corta del experimento, sin valor estadístico")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("La comparación mide tiempos con la RTX y CUDA no está disponible.")
    escala, limite = (40, 40) if a.rapido else (1, None)
    destino = SALIDA.with_name("arreglos_rapido") if a.rapido else SALIDA
    destino.mkdir(parents=True, exist_ok=True)
    modelos()
    items = cargar_items(limite)
    muestra = muestra_tiempo(items, 2 if a.rapido else 40)
    orden = ["directa", "mixta_refino"] if a.rapido else list(ARREGLOS)
    resultados = {n: evaluar_arreglo(n, items, muestra, destino, escala) for n in orden}
    svm = evaluar_svm(items, muestra, destino)
    salida = tabla(resultados, svm)
    (destino / "tabla.json").write_text(json.dumps(salida, indent=1, default=float), encoding="utf-8")
    print("MAS EFICIENTE:", salida["mas_eficiente"], "| con RTX:", salida["mas_eficiente_gpu"], flush=True)
    for f in salida["arreglos"]:
        print(f"{f['arreglo']:16s} exact {f['exactitud']:.3f} abierto {f['puntaje_abierto']:.3f} "
              f"auroc {f['auroc']:.3f} cpu {f['tiempo_ms_mediana']:.0f}/{f['tiempo_ms_p95']:.0f} ms "
              f"rtx {f['tiempo_gpu_ms_mediana']:.0f}/{f['tiempo_gpu_ms_p95']:.0f} ms", flush=True)
    print("svm", salida["svm"]["exactitud"], salida["svm"]["tiempo_ms_mediana"], flush=True)


if __name__ == "__main__":
    main()
