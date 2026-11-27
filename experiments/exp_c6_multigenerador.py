"""Entrena en la RTX con varios generadores y canales, apartando uno cada vez."""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed

from experiments.comun_cascada import DIR_MODELOS, RUTA_DESCRIPTOR, RUTA_RECUPERADOR, fila_resultado, metricas
from src.cascada import geometria as g
from src.cascada.cadena import recuperar
from src.cascada.decisor import DESCONOCIDA, INDETERMINADA, Decisor
from src.cascada.descriptor import (Descriptor, aumentar, calcular_huellas, cargar_descriptor, guardar_descriptor,
                                    imagen_densidad, perdida_contrastiva)
from src.cascada.detector import snr_ciega_db
from src.cascada.pipeline import calidad_de, guardar_paquete
from src.cascada.recuperador import cargar_recuperador

# ─────────────────────────────  Config  ──────────────────────────────
BANCO = Path("data/banco/base")
DECISOR_BASE = Path("results/cascada/arreglos/decisores/red_refino.joblib")
CADENA = "red"
NOMBRES = list(g.catalogo())
TRABAJOS = 10
PASOS = 8000
LOTE = 128
MAX_FONDO = 4800
FAMILIAS_GENERADOR = {"repo": ["repo"], "procedural": ["procedural"], "frecuencia": ["frecuencia"],
                      "analogico": ["analogico"], "gnuradio": ["gr_polifase", "gr_fir_mmse"]}
FAMILIAS_CANAL = {
    "multitrayecto": ["proyecto", "mp_corto", "mp_largo", "gr_multitrayecto"],
    "desvanecimiento": ["plano_lento", "tdl_rayleigh", "tdl_rician", "gr_selectivo", "gr_dinamico"],
    "sincronia": ["ruido_fase", "deriva_portadora", "deriva_reloj"],
    "hardware": ["iq_dc", "amplificador", "cuantizacion", "receptor_barato", "gr_imperfecciones"],
    "otros": ["interferencia", "impulsivo"],
}
_C = {}


# ───────────────────────────  Recuperación  ──────────────────────────
def recuperador():
    if "r" not in _C:
        torch.set_num_threads(1)
        _C["r"] = cargar_recuperador(RUTA_RECUPERADOR)
    return _C["r"]


def imagen_y_calidad(iq, sps):
    x = np.asarray(iq, dtype=np.complex128)
    r = recuperar(x, CADENA, recuperador(), sps=sps)
    if r is None:
        return None
    imagen = (np.clip(imagen_densidad(r["simbolos"]) / 1.5, 0, 1) * 255).astype(np.uint8)
    return imagen, calidad_de(r, snr_ciega_db(x))[0], float(r["sps_origen"])


def vista_pool(base, m):
    iq = np.fromfile(base / "iq.bin", np.complex64, count=m["n"], offset=m["offset"])
    return imagen_y_calidad(iq, None if m["tipo"].startswith("calibra") else m["sps"])


def vista_banco(f):
    return imagen_y_calidad(np.fromfile(BANCO / f["archivo"], np.complex64), None)


def empaquetar(salidas):
    n = len(salidas)
    imagenes = np.zeros((n, 2, 64, 64), np.uint8)
    calidad = np.full((n, 3), np.nan)
    sps = np.full(n, np.nan)
    valido = np.zeros(n, bool)
    for i, s in enumerate(salidas):
        if s is not None:
            imagenes[i], calidad[i], sps[i], valido[i] = s[0], s[1], s[2], True
    return {"imagenes": imagenes, "calidad": calidad, "sps": sps, "valido": valido}


def en_cache(ruta, crear):
    if ruta.exists():
        with np.load(ruta) as datos:
            return {k: datos[k] for k in datos.files}
    datos = crear()
    np.savez(ruta, **datos)
    return datos


def leer_jsonl(ruta):
    return [json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines()]


def cargar_datos(base, limite):
    meta = leer_jsonl(base / "meta.jsonl")
    pool = en_cache(base / "imagenes_pool.npz", lambda: empaquetar(
        Parallel(n_jobs=TRABAJOS, batch_size=16)(delayed(vista_pool)(base, m) for m in meta)))
    banco = [f for f in leer_jsonl(BANCO / "manifiesto.jsonl") if f["grupo"] != "validacion"][::limite]
    prueba = en_cache(base / "imagenes_banco.npz", lambda: empaquetar(
        Parallel(n_jobs=TRABAJOS, batch_size=8)(delayed(vista_banco)(f) for f in banco)))
    print("imágenes listas", len(meta), len(banco), flush=True)
    return meta, pool, banco, prueba


# ───────────────────────────  Particiones  ───────────────────────────
def particiones():
    salida = {f"gen_{k}": {"generadores": v, "canales": []} for k, v in FAMILIAS_GENERADOR.items()}
    salida.update({f"canal_{k}": {"generadores": [], "canales": v} for k, v in FAMILIAS_CANAL.items()})
    return salida


def apartada(fila, particion):
    return fila["generador"] in particion["generadores"] or fila["canal"] in particion["canales"]


# ───────────────────────────  Entrenamiento  ─────────────────────────
def armar_geometrias(meta, pool, permitidas):
    filas = [i for i, m in enumerate(meta) if m["tipo"] == "geo"]
    n_geo = max(meta[i]["indice"] for i in filas) + 1
    n_vistas = max(meta[i]["vista"] for i in filas) + 1
    x = torch.zeros((n_geo, n_vistas, 2, 64, 64), dtype=torch.uint8)
    mascara = torch.zeros((n_geo, n_vistas), dtype=torch.bool)
    forma = torch.zeros((n_geo, 2))
    for i in filas:
        m = meta[i]
        x[m["indice"], m["vista"]] = torch.from_numpy(pool["imagenes"][i])
        mascara[m["indice"], m["vista"]] = bool(pool["valido"][i] and permitidas[i])
        forma[m["indice"]] = torch.tensor([m["puntos"], m["anillos"]])
    return x, mascara, forma


def pares(x, mascara, forma, indices):
    fila = torch.as_tensor(indices)
    azar = torch.rand(mascara[fila].shape)
    azar[~mascara[fila]] = 2.0
    orden = azar.argsort(1)
    v1 = x[fila, orden[:, 0]].float() / 255 * 1.5
    v2 = x[fila, orden[:, 1]].float() / 255 * 1.5
    return aumentar(v1), aumentar(v2), forma[fila]


def perdida_validacion(modelo, x, mascara, forma, indices, disp):
    modelo.eval()
    total, n = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(indices) - LOTE + 1, LOTE):
            v1, v2, _ = pares(x, mascara, forma, indices[i:i + LOTE])
            total += perdida_contrastiva(modelo(v1.to(disp))[0], modelo(v2.to(disp))[0]).item()
            n += 1
    modelo.train()
    return total / max(n, 1)


def entrenar(x, mascara, forma, pasos, semilla, disp):
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    utiles = np.flatnonzero(mascara.sum(1).numpy() >= 2)
    rng.shuffle(utiles)
    corte = max(LOTE, len(utiles) // 10) if len(utiles) > 2 * LOTE else len(utiles) // 2
    validacion, entrenamiento = utiles[:corte], utiles[corte:]
    modelo = Descriptor().to(disp)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    mejor, estado, paso = math.inf, None, 0
    tamano = min(LOTE, len(entrenamiento))
    while paso < pasos:
        orden = rng.permutation(entrenamiento)
        for i in range(0, len(orden) - tamano + 1, tamano):
            if paso >= pasos:
                break
            for grupo in optimizador.param_groups:
                grupo["lr"] = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * paso / pasos))
            v1, v2, real = pares(x, mascara, forma, orden[i:i + tamano])
            real = real.to(disp)
            z1, g1 = modelo(v1.to(disp))
            z2, g2 = modelo(v2.to(disp))
            perdida = perdida_contrastiva(z1, z2) + 2.0 * (torch.nn.functional.smooth_l1_loss(g1, real)
                                                           + torch.nn.functional.smooth_l1_loss(g2, real))
            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            optimizador.step()
            paso += 1
            if paso % 500 == 0 or paso == pasos:
                valor = perdida_validacion(modelo, x, mascara, forma, validacion, disp)
                if valor < mejor:
                    mejor, estado = valor, {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
    modelo.load_state_dict(estado)
    return modelo.eval(), mejor


# ─────────────────────────────  Decisor  ─────────────────────────────
def a_flotante(imagenes):
    return imagenes.astype(np.float32) / 255 * 1.5


def ajustar_decisor(modelo, meta, pool, permitidas, semilla):
    rng = np.random.default_rng(semilla)
    util = pool["valido"] & permitidas
    tipo = np.array([m["tipo"] for m in meta])
    etiqueta = np.array([m["etiqueta"] for m in meta])
    huellas = lambda i: calcular_huellas(modelo, a_flotante(pool["imagenes"][i]))[0]
    catalogo = {n: huellas(np.flatnonzero(util & (tipo == "catalogo") & (etiqueta == n))) for n in NOMBRES}
    fondo = np.flatnonzero(util & (tipo == "geo"))
    fondo = rng.choice(fondo, min(MAX_FONDO, len(fondo)), replace=False)
    decisor = Decisor().ajustar(huellas(fondo), catalogo)
    calibra = np.flatnonzero(util & np.isin(tipo, ["calibra", "calibra_geo"]))
    return decisor.calibrar(huellas(calibra), etiqueta[calibra].tolist(), pool["calidad"][calibra])


# ─────────────────────────────  Evaluación  ──────────────────────────
def filas_evaluacion(modelo, decisor, banco, prueba, indices):
    validos = [i for i in indices if prueba["valido"][i]]
    z = calcular_huellas(modelo, a_flotante(prueba["imagenes"][validos]))[0]
    p = decisor.probabilidades(z, prueba["calidad"][validos])
    por_archivo = dict(zip(validos, p))
    nombres = decisor.nombres
    filas = []
    for i in indices:
        f = banco[i]
        item = {"real": f["modulacion"], "etiqueta": f["etiqueta"], "snr": f["snr_db"], "sps": f["sps"]}
        if i in por_archivo:
            q = por_archivo[i]
            k = int(q.argmax())
            salida = {"modulacion": nombres[k] if q[k] >= decisor.umbral else INDETERMINADA, "confianza": float(q[k]),
                      "probabilidades": {n: float(v) for n, v in zip(nombres, q)},
                      "etapas": {"sps_origen": float(prueba["sps"][i])}}
        else:
            salida = {"modulacion": INDETERMINADA, "confianza": None, "probabilidades": {}, "etapas": {}}
        filas.append({**fila_resultado(item, salida, NOMBRES), "generador": f["generador"], "canal": f["canal"]})
    return filas


def resumen(filas):
    m = metricas([f for f in filas if f["snr"] >= 10], NOMBRES)
    return {"archivos": m["n"], "exactitud": m["exactitud_cerrada"], "correctas_salida_real": m["exactitud_abierta"],
            "desconocidas_detectadas": m["rechazo_desconocidas"], "falsa_aceptacion": m["falsa_aceptacion"],
            "auroc": m["auroc_desconocidas"], "ece": m.get("ece"),
            "puntaje_abierto": (m["exactitud_abierta"] + m["rechazo_desconocidas"]) / 2}


def matriz(filas):
    altas = [f for f in filas if f["snr"] >= 10 and f["etiqueta"] in NOMBRES]
    celdas = {}
    for f in altas:
        celdas.setdefault(f["generador"], {}).setdefault(f["canal"], []).append(f["cerrada"] == f["etiqueta"])
    return {a: {b: float(np.mean(v)) for b, v in fila.items()} for a, fila in celdas.items()}


def promedio(bloques, clave):
    return {k: float(np.mean([b[clave][k] for b in bloques])) for k in bloques[0][clave] if k != "archivos"}


# ──────────────────────────────  Etapas  ─────────────────────────────
def correr_particion(nombre, particion, semilla, meta, pool, banco, prueba, base_filas, salida, pasos, disp):
    ruta = salida / f"{nombre}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    inicio = time.time()
    permitidas = np.array([not apartada(m, particion) for m in meta])
    x, mascara, forma = armar_geometrias(meta, pool, permitidas)
    modelo, valor = entrenar(x, mascara, forma, pasos, semilla, disp)
    decisor = ajustar_decisor(modelo, meta, pool, permitidas, semilla)
    indices = [i for i, f in enumerate(banco) if apartada(f, particion)]
    filas = filas_evaluacion(modelo, decisor, banco, prueba, indices)
    referencia = [base_filas[i] for i in indices]
    datos = {"particion": nombre, "apartado": particion, "val": valor, "umbral": decisor.umbral,
             "vistas_entrenamiento": int(permitidas.sum()), "minutos": (time.time() - inicio) / 60,
             "multi": resumen(filas), "referencia": resumen(referencia), "filas": filas}
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    print(nombre, f"{datos['minutos']:.1f} min", "multi", round(datos["multi"]["exactitud"], 3),
          "auroc", round(datos["multi"]["auroc"], 3), "| referencia", round(datos["referencia"]["exactitud"], 3),
          "auroc", round(datos["referencia"]["auroc"], 3), flush=True)
    return datos


def modelo_final(meta, pool, destino, pasos, disp):
    permitidas = np.ones(len(meta), bool)
    x, mascara, forma = armar_geometrias(meta, pool, permitidas)
    modelo, valor = entrenar(x, mascara, forma, pasos, 2026112099, disp)
    decisor = ajustar_decisor(modelo, meta, pool, permitidas, 2026112099)
    destino.mkdir(parents=True, exist_ok=True)
    guardar_descriptor(modelo.cpu(), destino / "descriptor.pt", {"val": valor, "pasos": pasos, "cadena": CADENA})
    shutil.copyfile(RUTA_RECUPERADOR, destino / "recuperador.pt")
    guardar_paquete(destino / "paquete.joblib", decisor, CADENA, "recuperador.pt", "descriptor.pt",
                    {"entrenamiento": "multigenerador", "vistas": len(meta)})
    print("modelo final guardado", destino, flush=True)


def paquete_referencia():
    destino = DIR_MODELOS / "paquete_red.joblib"
    if not destino.exists():
        guardar_paquete(destino, joblib.load(DECISOR_BASE), CADENA, RUTA_RECUPERADOR.name, RUTA_DESCRIPTOR.name,
                        {"entrenamiento": "procedural"})


# ──────────────────────────────  Entry  ──────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rapido", action="store_true")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Este entrenamiento necesita la RTX y CUDA no está disponible.")
    disp = torch.device("cuda")
    sufijo = "_rapido" if a.rapido else ""
    base = Path("data/cascada_multi" + sufijo)
    salida = Path("results/cascada/multi" + sufijo)
    salida.mkdir(parents=True, exist_ok=True)
    pasos = 30 if a.rapido else PASOS
    meta, pool, banco, prueba = cargar_datos(base, 40 if a.rapido else 1)
    torch.set_num_threads(4)
    paquete_referencia()
    referencia = cargar_descriptor(RUTA_DESCRIPTOR).to(disp)
    base_filas = filas_evaluacion(referencia, joblib.load(DECISOR_BASE), banco, prueba, list(range(len(banco))))
    planes = list(particiones().items())
    planes = planes[:1] + planes[5:6] if a.rapido else planes
    bloques = [correr_particion(n, c, 2026112020 + i, meta, pool, banco, prueba, base_filas, salida, pasos, disp)
               for i, (n, c) in enumerate(planes)]
    grupos = {"generadores": [b for b in bloques if b["particion"].startswith("gen_")],
              "canales": [b for b in bloques if b["particion"].startswith("canal_")]}
    total = {"referencia_todo_el_banco": resumen(base_filas), "matriz_referencia": matriz(base_filas),
             "matriz_generador_apartado": matriz([f for b in grupos["generadores"] for f in b["filas"]]),
             "matriz_canal_apartado": matriz([f for b in grupos["canales"] for f in b["filas"]]),
             "promedios": {k: {"multi": promedio(v, "multi"), "referencia": promedio(v, "referencia")}
                           for k, v in grupos.items() if v},
             "particiones": [{k: b[k] for k in ("particion", "apartado", "minutos", "vistas_entrenamiento", "umbral",
                                                "multi", "referencia")} for b in bloques]}
    (salida / "resumen.json").write_text(json.dumps(total, indent=1), encoding="utf-8")
    print(json.dumps(total["promedios"], indent=1), flush=True)
    modelo_final(meta, pool, Path("results/cascada/modelos_multi" + sufijo), pasos, disp)


if __name__ == "__main__":
    main()
