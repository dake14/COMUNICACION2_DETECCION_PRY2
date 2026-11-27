"""Etapa 2: especialistas por vista y fusión, evaluados dejando fuera pares de generadores.

    .venv/Scripts/python.exe -m experiments.exp_f1_especialistas --etapa todo
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from src.banco.modulaciones import DESCONOCIDA, catalogo_digital, modulaciones_digitales
from src.familias.enrutador import Enrutador, Fusion
from src.familias.especialistas import entrenar, log_probabilidades
from src.familias.vistas import VISTAS, vistas

# ─────────────────────────────  Config  ──────────────────────────────
ENTRENO = Path("data/banco/digitales_entreno")
EVALUACION = Path("data/banco/digitales")
SALIDA = Path("results/familias")
MODELO = SALIDA / "modelo"
VENTANAS_ENTRENO = 4
FRACCION_MIN = 0.3
MUESTRAS_MIN = 512
SEMILLA = 2026112510
PARTE_FUSION = 0.2
PLIEGUES = {"repo+gr_fir_mmse": ("repo", "gr_fir_mmse"), "procedural+gr_polifase": ("procedural", "gr_polifase"),
            "frecuencia+analogico": ("frecuencia", "analogico")}
CATALOGO = catalogo_digital()
CLASES = list(CATALOGO)
FAMILIA_DE = {k: m.familia for k, m in CATALOGO.items()}
FAMILIA_REAL = {k: m.familia for k, (m, _, _) in modulaciones_digitales().items()}


# ───────────────────────────────  Datos  ─────────────────────────────
def manifiesto(base):
    return [json.loads(x) for x in (base / "manifiesto.jsonl").read_text(encoding="utf-8").splitlines()]


def _vistas_archivo(base, archivo, ventanas):
    x = np.fromfile(base / archivo, np.complex64)
    if ventanas == 1:
        return [vistas(x)[0]]
    rng = np.random.default_rng([SEMILLA, zlib.crc32(archivo.encode())])
    salida = []
    for _ in range(ventanas):
        largo = max(min(len(x), MUESTRAS_MIN), int(len(x) * rng.uniform(FRACCION_MIN, 1.0)))
        inicio = int(rng.integers(0, len(x) - largo + 1))
        salida.append(vistas(x[inicio:inicio + largo])[0])
    return salida


def extraer(base, ventanas, trabajos=12):
    cache = base / f"vistas_f1_{ventanas}.npz"
    if cache.exists():
        d = np.load(cache)
        return {v: d[v] for v in VISTAS}, d["archivo"]
    filas = manifiesto(base)
    inicio = time.time()
    partes = Parallel(n_jobs=trabajos, batch_size=16)(delayed(_vistas_archivo)(base, f["archivo"], ventanas)
                                                       for f in filas)
    indice = np.concatenate([np.full(len(p), i) for i, p in enumerate(partes)])
    x = {v: np.stack([w[v] for p in partes for w in p]).astype(np.float16) for v in VISTAS}
    np.savez(cache, archivo=indice, **x)
    print("vistas", base, len(indice), f"{(time.time() - inicio) / 60:.1f} min", flush=True)
    return x, indice


# ─────────────────────────────  Entrenar  ────────────────────────────
def entrenar_enrutador(x, indice, filas, generadores, disp, pasos, semilla):
    rng = np.random.default_rng(semilla)
    archivos = np.array([i for i, f in enumerate(filas) if f["generador"] in generadores])
    archivos = rng.permutation(archivos)
    corte = int(len(archivos) * PARTE_FUSION)
    de_fusion, de_especialistas = archivos[:corte], archivos[corte:]
    y = np.array([CLASES.index(f["modulacion"]) for f in filas])[indice]
    m_esp, m_fus = np.isin(indice, de_especialistas), np.isin(indice, de_fusion)
    especialistas, validacion = {}, {}
    for k, v in enumerate(VISTAS):
        especialistas[v], validacion[v] = entrenar(x[v][m_esp], y[m_esp], len(CLASES), disp, pasos, semilla=semilla + k)
    logp = {v: log_probabilidades(especialistas[v], x[v][m_fus]) for v in VISTAS}
    fusion = Fusion(CLASES, FAMILIA_DE).ajustar(logp, y[m_fus])
    return Enrutador(especialistas, fusion), validacion


# ─────────────────────────────  Evaluar  ─────────────────────────────
def filas_evaluacion(enrutador, x, filas, elegidos):
    logp = enrutador.log_probabilidades({v: x[v][elegidos] for v in VISTAS})
    p = enrutador.fusion.probabilidades(logp)
    pf = enrutador.fusion.por_familia(p)
    salida = []
    for j, i in enumerate(elegidos):
        f = filas[i]
        fila = {"archivo": f["archivo"], "modulacion": f["modulacion"], "etiqueta": f["etiqueta"], "grupo": f["grupo"],
                "familia": FAMILIA_REAL[f["modulacion"]], "generador": f["generador"], "canal": f["canal"],
                "snr": f["snr_db"], "sps": f["sps"], "clase": CLASES[int(p[j].argmax())],
                "confianza": float(p[j].max()), "familia_predicha": enrutador.fusion.familias[int(pf[j].argmax())],
                "confianza_familia": float(pf[j].max())}
        fila.update({f"clase_{v}": CLASES[int(logp[v][j].argmax())] for v in VISTAS})
        salida.append(fila)
    return salida


def _exactitud(filas, clave):
    return round(float(np.mean([f[clave] == f["modulacion"] for f in filas])), 4) if filas else None


def _exactitud_familia(filas, clave):
    if not filas:
        return None
    return round(float(np.mean([FAMILIA_DE.get(f[clave], f[clave]) == f["familia"] for f in filas])), 4)


def resumen(filas):
    conocidas = [f for f in filas if f["grupo"] == "catalogo"]
    altas = [f for f in conocidas if f["snr"] >= 10]
    salida = {"archivos_conocidos": len(conocidas)}
    for nombre, grupo in (("todas", conocidas), ("snr>=10", altas)):
        salida[nombre] = {"clase": {"fusion": _exactitud(grupo, "clase"),
                                    **{v: _exactitud(grupo, f"clase_{v}") for v in VISTAS}},
                          "familia": {"fusion": _exactitud_familia(grupo, "familia_predicha"),
                                      **{v: _exactitud_familia(grupo, f"clase_{v}") for v in VISTAS}}}
    salida["por_clase_snr>=10"] = {c: _exactitud([f for f in altas if f["modulacion"] == c], "clase") for c in CLASES}
    salida["por_familia_snr>=10"] = {
        fam: _exactitud_familia([f for f in altas if f["familia"] == fam], "familia_predicha")
        for fam in sorted(set(FAMILIA_DE.values()))}
    salida["por_snr"] = {str(s): _exactitud([f for f in conocidas if f["snr"] == s], "clase")
                         for s in sorted({f["snr"] for f in conocidas})}
    salida["por_canal_snr>=10"] = {c: _exactitud([f for f in altas if f["canal"] == c], "clase")
                                   for c in sorted({f["canal"] for f in altas})}
    confusion = {}
    for f in altas:
        if f["clase"] != f["modulacion"]:
            clave = f"{f['modulacion']}->{f['clase']}"
            confusion[clave] = confusion.get(clave, 0) + 1
    salida["confusiones_snr>=10"] = dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:25])
    desconocidas = [f for f in filas if f["etiqueta"] == DESCONOCIDA and f["snr"] >= 10]
    salida["desconocidas_familia_snr>=10"] = {
        m: {fam: round(float(np.mean([f["familia_predicha"] == fam for f in grupo])), 3)
            for fam in sorted({f["familia_predicha"] for f in grupo})}
        for m in sorted({f["modulacion"] for f in desconocidas})
        for grupo in [[f for f in desconocidas if f["modulacion"] == m]]}
    return salida


# ─────────────────────────────  Etapas  ──────────────────────────────
def pliegues(disp, pasos):
    x, indice = extraer(ENTRENO, VENTANAS_ENTRENO)
    xe, indice_e = extraer(EVALUACION, 1)
    filas, filas_e = manifiesto(ENTRENO), manifiesto(EVALUACION)
    todas, detalle = [], {}
    for k, (nombre, fuera) in enumerate(PLIEGUES.items()):
        inicio = time.time()
        dentro = sorted({f["generador"] for f in filas} - set(fuera))
        enrutador, validacion = entrenar_enrutador(x, indice, filas, dentro, disp, pasos, SEMILLA + 100 * k)
        elegidos = np.array([j for j, f in enumerate(filas_e) if f["generador"] in fuera])
        filas_pliegue = filas_evaluacion(enrutador, xe, filas_e, elegidos[np.argsort(indice_e[elegidos])])
        todas += filas_pliegue
        detalle[nombre] = {"validacion_interna": validacion, **resumen(filas_pliegue)}
        r = detalle[nombre]["snr>=10"]
        print(nombre, f"{(time.time() - inicio) / 60:.1f} min", "clase", r["clase"], "familia", r["familia"]["fusion"],
              flush=True)
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "filas_pliegues.json").write_text(json.dumps(todas), encoding="utf-8")
    total = {"pasos": pasos, "fuera_de_generador": resumen(todas), "pliegues": detalle}
    (SALIDA / "resumen.json").write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(total["fuera_de_generador"]["snr>=10"], indent=1), flush=True)


def final(disp, pasos):
    x, indice = extraer(ENTRENO, VENTANAS_ENTRENO)
    filas = manifiesto(ENTRENO)
    enrutador, validacion = entrenar_enrutador(x, indice, filas, sorted({f["generador"] for f in filas}), disp, pasos,
                                               SEMILLA + 999)
    enrutador.guardar(MODELO)
    (MODELO / "info.json").write_text(json.dumps({"clases": CLASES, "familias": FAMILIA_DE, "pasos": pasos,
                                                  "validacion_interna": validacion}, indent=1), encoding="utf-8")
    print("modelo final", MODELO, validacion, flush=True)


# ──────────────────────────────  Entry  ──────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", choices=("extraer", "pliegues", "final", "todo"), default="todo")
    p.add_argument("--pasos", type=int, default=4000)
    a = p.parse_args()
    if a.etapa == "extraer":
        extraer(ENTRENO, VENTANAS_ENTRENO)
        extraer(EVALUACION, 1)
        return
    if not torch.cuda.is_available():
        raise SystemExit("Este entrenamiento necesita la RTX y CUDA no está disponible.")
    disp = torch.device("cuda")
    if a.etapa in ("pliegues", "todo"):
        pliegues(disp, a.pasos)
    if a.etapa in ("final", "todo"):
        final(disp, a.pasos)


if __name__ == "__main__":
    main()
