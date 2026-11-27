"""Compara arreglos de la CNN frente a generadores y familias de canal que no vio al entrenar.

    .venv/Scripts/python.exe -m experiments.exp_c8_arreglos_cnn --etapa cribado
    .venv/Scripts/python.exe -m experiments.exp_c8_arreglos_cnn --etapa completo --variante <ganadora>
    .venv/Scripts/python.exe -m experiments.exp_c8_arreglos_cnn --etapa final --variante <ganadora>
"""
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
import torch.nn.functional as F
from joblib import Parallel, delayed

from experiments.comun_cascada import RUTA_DESCRIPTOR, RUTA_RECUPERADOR, fila_resultado, metricas
from src.cascada import geometria as g
from src.cascada.cadena import recuperar
from src.cascada.decisor import INDETERMINADA, Decisor
from src.cascada.descriptor import (Descriptor, aumentar, calcular_huellas, cargar_descriptor, guardar_descriptor,
                                    imagen_densidad, perdida_contrastiva)
from src.cascada.descriptor_puntos import PUNTOS, DescriptorPuntos, aumentar_puntos, entrada_puntos
from src.cascada.descriptor_variantes import DescriptorAdversario, DescriptorAncho
from src.cascada.detector import snr_ciega_db
from src.cascada.pipeline import calidad_de, guardar_paquete
from src.cascada.recuperador import cargar_recuperador

# ─────────────────────────────  Config  ──────────────────────────────
POOL = Path("data/cascada_variado")
BANCOS = {"base": Path("data/banco/base"), "canales2": Path("data/banco/canales2")}
RECUPERADOR_ROBUSTO = Path("results/cascada/modelos_robustos/recuperador.pt")
DECISOR_REFERENCIA = Path("results/cascada/arreglos/decisores/red_refino.joblib")
SALIDA = Path("results/cascada/arreglos_cnn")
DESTINO_FINAL = Path("results/cascada/modelos_robustos")
CADENA = "red"
NOMBRES = list(g.catalogo())
SIMBOLOS = 1024
TRABAJOS = 10
LOTE = 128
MAX_FONDO = 4800
PESO_DOMINIO = 0.3
VARIANTES = ("imagen", "ancha", "puntos", "adversaria", "multivista")
ENTRENADAS = ("imagen", "ancha", "puntos", "adversaria")
FAMILIAS_GENERADOR = {"repo": ["repo"], "procedural": ["procedural"], "frecuencia": ["frecuencia"],
                      "analogico": ["analogico"], "gnuradio": ["gr_polifase", "gr_fir_mmse"]}
FAMILIAS_CANAL = {
    "multitrayecto": ["proyecto", "mp_corto", "mp_largo", "gr_multitrayecto", "dos_rayos", "ecos_largos"],
    "desvanecimiento": ["plano_lento", "tdl_rayleigh", "tdl_rician", "gr_selectivo", "gr_dinamico"],
    "lte": ["lte_epa", "lte_eva", "lte_etu"],
    "desvanecimiento_extra": ["nakagami", "sombra", "watterson"],
    "sincronia": ["ruido_fase", "deriva_portadora", "deriva_reloj"],
    "hardware": ["iq_dc", "amplificador", "cuantizacion", "receptor_barato", "gr_imperfecciones"],
    "interferencia": ["interferencia", "impulsivo", "int_adyacente", "int_cocanal", "int_barrido", "int_pulsos"],
}
CRIBADO = ("gen_gnuradio", "gen_analogico", "canal_desvanecimiento", "canal_lte")
DOMINIOS = ["ideal", *FAMILIAS_CANAL]
_C = {}


# ───────────────────────────  Recuperación  ──────────────────────────
def recuperador(ruta):
    if _C.get("ruta") != ruta:
        torch.set_num_threads(1)
        _C.update(ruta=ruta, r=cargar_recuperador(ruta))
    return _C["r"]


def extraer(iq, sps, ruta):
    x = np.asarray(iq, dtype=np.complex128)
    r = recuperar(x, CADENA, recuperador(ruta), sps=sps)
    if r is None:
        return None
    z = r["simbolos"][:SIMBOLOS]
    imagen = (np.clip(imagen_densidad(r["simbolos"]) / 1.5, 0, 1) * 255).astype(np.uint8)
    return imagen, calidad_de(r, snr_ciega_db(x))[0], float(r["sps_origen"]), z.astype(np.complex64)


def vista_pool(m, ruta):
    iq = np.fromfile(POOL / "iq.bin", np.complex64, count=m["n"], offset=m["offset"])
    return extraer(iq, None if m["tipo"].startswith("calibra") else m["sps"], ruta)


def vista_banco(base, f, ruta):
    return extraer(np.fromfile(base / f["archivo"], np.complex64), None, ruta)


def empaquetar(salidas):
    n = len(salidas)
    datos = {"imagenes": np.zeros((n, 2, 64, 64), np.uint8), "calidad": np.full((n, 3), np.nan),
             "sps": np.full(n, np.nan), "valido": np.zeros(n, bool),
             "simbolos": np.zeros((n, SIMBOLOS), np.complex64), "cuenta": np.zeros(n, np.int32)}
    for i, s in enumerate(salidas):
        if s is not None:
            datos["imagenes"][i], datos["calidad"][i], datos["sps"][i] = s[0], s[1], s[2]
            datos["simbolos"][i, :len(s[3])], datos["cuenta"][i], datos["valido"][i] = s[3], len(s[3]), True
    return datos


def en_cache(ruta, crear):
    if ruta.exists():
        with np.load(ruta) as datos:
            return {k: datos[k] for k in datos.files}
    datos = crear()
    np.savez(ruta, **datos)
    return datos


def leer_jsonl(ruta):
    return [json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines()]


def cargar_pool(ruta):
    meta = leer_jsonl(POOL / "meta.jsonl")
    datos = en_cache(POOL / "datos_c8.npz", lambda: empaquetar(
        Parallel(n_jobs=TRABAJOS, batch_size=16)(delayed(vista_pool)(m, ruta) for m in meta)))
    print("pool listo", len(meta), int(datos["valido"].sum()), flush=True)
    return meta, datos


def cargar_bancos(ruta, sufijo):
    filas, partes = [], []
    for nombre, base in BANCOS.items():
        if not (base / "manifiesto.jsonl").exists():
            continue
        banco = [f for f in leer_jsonl(base / "manifiesto.jsonl") if f["grupo"] != "validacion"]
        partes.append(en_cache(base / f"datos_c8{sufijo}.npz", lambda: empaquetar(
            Parallel(n_jobs=TRABAJOS, batch_size=8)(delayed(vista_banco)(base, f, ruta) for f in banco))))
        filas += [{**f, "banco": nombre} for f in banco]
    datos = {k: np.concatenate([p[k] for p in partes]) for k in partes[0]}
    print("bancos listos", sufijo or "robusto", len(filas), int(datos["valido"].sum()), flush=True)
    return filas, datos


# ───────────────────────────  Particiones  ───────────────────────────
def familia(canal):
    return next((k for k, v in FAMILIAS_CANAL.items() if canal in v), "ideal")


def particiones():
    salida = {f"gen_{k}": {"generadores": v, "canales": []} for k, v in FAMILIAS_GENERADOR.items()}
    salida.update({f"canal_{k}": {"generadores": [], "canales": v} for k, v in FAMILIAS_CANAL.items()})
    return salida


def apartada(fila, particion):
    return fila["generador"] in particion["generadores"] or fila["canal"] in particion["canales"]


# ─────────────────────────────  Entradas  ────────────────────────────
class Tensores:
    def __init__(self, datos, meta):
        self.imagenes = torch.from_numpy(datos["imagenes"])
        self.simbolos = torch.view_as_real(torch.from_numpy(datos["simbolos"]))
        self.cuenta = torch.from_numpy(datos["cuenta"]).clamp_min(1)
        self.dominio = torch.tensor([DOMINIOS.index(familia(m["canal"])) for m in meta])

    def imagen(self, i):
        return aumentar(self.imagenes[i].float() / 255 * 1.5)

    def puntos(self, i):
        s, n = self.simbolos[i], self.cuenta[i]
        idx = (torch.rand(len(i), PUNTOS) * n[:, None]).long()
        p = torch.gather(s, 1, idx[..., None].expand(-1, -1, 2))
        p = p - p.mean(1, keepdim=True)
        p = p / p.pow(2).sum(2).mean(1).clamp_min(1e-12).sqrt()[:, None, None]
        return aumentar_puntos(torch.stack((p[..., 0], p[..., 1], p.norm(dim=2)), 1))


def armar_geometrias(meta, datos, permitidas):
    filas = [i for i, m in enumerate(meta) if m["tipo"] == "geo"]
    n_geo = max(meta[i]["indice"] for i in filas) + 1
    n_vistas = max(meta[i]["vista"] for i in filas) + 1
    vistas = torch.full((n_geo, n_vistas), -1, dtype=torch.long)
    forma = torch.zeros((n_geo, 2))
    for i in filas:
        m = meta[i]
        if datos["valido"][i] and permitidas[i]:
            vistas[m["indice"], m["vista"]] = i
        forma[m["indice"]] = torch.tensor([m["puntos"], m["anillos"]])
    return vistas, forma


def pares(vistas, indices):
    fila = torch.as_tensor(indices)
    v = vistas[fila]
    azar = torch.rand(v.shape)
    azar[v < 0] = 2.0
    orden = azar.argsort(1)
    return v.gather(1, orden[:, :1])[:, 0], v.gather(1, orden[:, 1:2])[:, 0]


# ─────────────────────────────  Modelos  ─────────────────────────────
def crear_modelo(variante):
    return {"imagen": Descriptor, "ancha": DescriptorAncho, "puntos": DescriptorPuntos,
            "adversaria": lambda: DescriptorAdversario(len(DOMINIOS))}[variante]()


def entrada(t, variante, i):
    return t.puntos(i) if variante == "puntos" else t.imagen(i)


def paso_perdida(modelo, variante, t, i1, i2, real, avance, disp):
    x1, x2 = entrada(t, variante, i1).to(disp), entrada(t, variante, i2).to(disp)
    if variante == "adversaria":
        alfa = 2 / (1 + math.exp(-10 * avance)) - 1
        z1, g1, d1 = modelo(x1, alfa)
        z2, g2, d2 = modelo(x2, alfa)
        dominio = PESO_DOMINIO * (F.cross_entropy(d1, t.dominio[i1].to(disp)) + F.cross_entropy(d2, t.dominio[i2].to(disp)))
    else:
        (z1, g1), (z2, g2), dominio = modelo(x1), modelo(x2), 0.0
    return (perdida_contrastiva(z1, z2) + 2.0 * (F.smooth_l1_loss(g1, real) + F.smooth_l1_loss(g2, real))
            + dominio)


def validar(modelo, variante, t, vistas, indices, disp):
    modelo.eval()
    total, n = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(indices) - LOTE + 1, LOTE):
            i1, i2 = pares(vistas, indices[i:i + LOTE])
            a = modelo(entrada(t, variante, i1).to(disp))[0]
            b = modelo(entrada(t, variante, i2).to(disp))[0]
            total += perdida_contrastiva(a, b).item()
            n += 1
    modelo.train()
    return total / max(n, 1)


def entrenar(variante, t, vistas, forma, pasos, semilla, disp):
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    utiles = np.flatnonzero((vistas >= 0).sum(1).numpy() >= 2)
    rng.shuffle(utiles)
    corte = max(LOTE, len(utiles) // 10) if len(utiles) > 2 * LOTE else len(utiles) // 2
    validacion, entrenamiento = utiles[:corte], utiles[corte:]
    if len(entrenamiento) == 0:
        raise ValueError("No hay geometrías con dos vistas válidas para entrenar.")
    tamano = min(LOTE, len(entrenamiento))
    modelo = crear_modelo(variante).to(disp)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    mejor, estado, paso = math.inf, None, 0
    while paso < pasos:
        orden = rng.permutation(entrenamiento)
        for i in range(0, len(orden) - tamano + 1, tamano):
            if paso >= pasos:
                break
            for grupo in optimizador.param_groups:
                grupo["lr"] = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * paso / pasos))
            k = orden[i:i + tamano]
            i1, i2 = pares(vistas, k)
            perdida = paso_perdida(modelo, variante, t, i1, i2, forma[torch.as_tensor(k)].to(disp), paso / pasos, disp)
            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            optimizador.step()
            paso += 1
            if paso % 500 == 0 or paso == pasos:
                valor = validar(modelo, variante, t, vistas, validacion, disp)
                if valor < mejor:
                    mejor, estado = valor, {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
    modelo.load_state_dict(estado)
    modelo = modelo.eval()
    return (modelo.descriptor() if variante == "adversaria" else modelo), mejor


# ─────────────────────────────  Huellas  ─────────────────────────────
def huellas(modelos, variante, datos, indices):
    indices = np.asarray(indices, dtype=int)
    if variante == "multivista":
        a = huellas(modelos, "imagen", datos, indices)
        b = huellas(modelos, "puntos", datos, indices)
        return np.concatenate((a, b), 1) / math.sqrt(2)
    if variante == "puntos":
        x = np.stack([entrada_puntos(datos["simbolos"][i, :datos["cuenta"][i]]) for i in indices]) \
            if len(indices) else np.zeros((0, 3, PUNTOS), np.float32)
        return calcular_huellas(modelos["puntos"], x)[0]
    return calcular_huellas(modelos[variante], datos["imagenes"][indices].astype(np.float32) / 255 * 1.5)[0]


def ajustar_decisor(modelos, variante, meta, datos, permitidas, semilla):
    rng = np.random.default_rng(semilla)
    util = datos["valido"] & permitidas
    tipo = np.array([m["tipo"] for m in meta])
    etiqueta = np.array([m["etiqueta"] for m in meta])
    h = lambda i: huellas(modelos, variante, datos, i)
    catalogo = {n: h(np.flatnonzero(util & (tipo == "catalogo") & (etiqueta == n))) for n in NOMBRES}
    fondo = np.flatnonzero(util & (tipo == "geo"))
    fondo = rng.choice(fondo, min(MAX_FONDO, len(fondo)), replace=False)
    decisor = Decisor().ajustar(h(fondo), catalogo)
    calibra = np.flatnonzero(util & np.isin(tipo, ["calibra", "calibra_geo"]))
    return decisor.calibrar(h(calibra), etiqueta[calibra].tolist(), datos["calidad"][calibra])


# ─────────────────────────────  Evaluación  ──────────────────────────
def filas_evaluacion(z_de, decisor, banco, datos, indices):
    validos = [i for i in indices if datos["valido"][i]]
    p = decisor.probabilidades(z_de(validos), datos["calidad"][validos]) if validos else []
    por_archivo = dict(zip(validos, p))
    filas = []
    for i in indices:
        f = banco[i]
        item = {"real": f["modulacion"], "etiqueta": f["etiqueta"], "snr": f["snr_db"], "sps": f["sps"]}
        if i in por_archivo:
            q = por_archivo[i]
            k = int(q.argmax())
            salida = {"modulacion": decisor.nombres[k] if q[k] >= decisor.umbral else INDETERMINADA,
                      "confianza": float(q[k]), "probabilidades": {n: float(v) for n, v in zip(decisor.nombres, q)},
                      "etapas": {"sps_origen": float(datos["sps"][i])}}
        else:
            salida = {"modulacion": INDETERMINADA, "confianza": None, "probabilidades": {}, "etapas": {}}
        filas.append({**fila_resultado(item, salida, NOMBRES), "generador": f["generador"], "canal": f["canal"],
                      "banco": f["banco"]})
    return filas


def resumen(filas):
    m = metricas([f for f in filas if f["snr"] >= 10], NOMBRES)
    todas = metricas(filas, NOMBRES)
    return {"archivos": m["n"], "exactitud": m["exactitud_cerrada"], "correctas_salida_real": m["exactitud_abierta"],
            "desconocidas_detectadas": m["rechazo_desconocidas"], "falsa_aceptacion": m["falsa_aceptacion"],
            "auroc": m["auroc_desconocidas"], "puntaje_abierto": (m["exactitud_abierta"] + m["rechazo_desconocidas"]) / 2,
            "exactitud_todas_snr": todas["exactitud_cerrada"]}


def por_canal(filas):
    grupos = {}
    for f in filas:
        if f["snr"] >= 10 and f["etiqueta"] in NOMBRES:
            grupos.setdefault(f["canal"], []).append(f["cerrada"] == f["etiqueta"])
    return {k: round(float(np.mean(v)), 3) for k, v in sorted(grupos.items())}


# ──────────────────────────────  Etapas  ─────────────────────────────
class Contexto:
    def __init__(self, disp):
        self.disp = disp
        ruta = RECUPERADOR_ROBUSTO if RECUPERADOR_ROBUSTO.exists() else RUTA_RECUPERADOR
        self.ruta_recuperador = ruta
        self.meta, self.pool = cargar_pool(ruta)
        self.t = Tensores(self.pool, self.meta)
        self.banco, self.prueba = cargar_bancos(ruta, "")
        self.referencia = None

    def filas_referencia(self):
        if self.referencia is None:
            banco, datos = cargar_bancos(RUTA_RECUPERADOR, "_ref")
            modelo = cargar_descriptor(RUTA_DESCRIPTOR).to(self.disp)
            z = lambda i: calcular_huellas(modelo, datos["imagenes"][i].astype(np.float32) / 255 * 1.5)[0]
            self.referencia = filas_evaluacion(z, joblib.load(DECISOR_REFERENCIA), banco, datos, list(range(len(banco))))
        return self.referencia


def correr(ctx, variantes, nombre, particion, semilla, pasos):
    permitidas = np.array([not apartada(m, particion) for m in ctx.meta])
    indices = [i for i, f in enumerate(ctx.banco) if apartada(f, particion)]
    if not indices:
        return []
    vistas, forma = armar_geometrias(ctx.meta, ctx.pool, permitidas)
    modelos, salidas = {}, []
    for variante in variantes:
        ruta = SALIDA / variante / f"{nombre}.json"
        if ruta.exists():
            salidas.append(json.loads(ruta.read_text(encoding="utf-8")))
            continue
        inicio = time.time()
        necesarias = ("imagen", "puntos") if variante == "multivista" else (variante,)
        valor = None
        for v in necesarias:
            if v not in modelos:
                modelos[v], valor = entrenar(v, ctx.t, vistas, forma, pasos, semilla, ctx.disp)
        decisor = ajustar_decisor(modelos, variante, ctx.meta, ctx.pool, permitidas, semilla)
        filas = filas_evaluacion(lambda i: huellas(modelos, variante, ctx.prueba, i), decisor, ctx.banco, ctx.prueba,
                                 indices)
        referencia = [ctx.filas_referencia()[i] for i in indices]
        datos = {"variante": variante, "particion": nombre, "apartado": particion, "val": valor, "pasos": pasos,
                 "umbral": decisor.umbral, "minutos": (time.time() - inicio) / 60, "resultado": resumen(filas),
                 "referencia": resumen(referencia), "por_canal": por_canal(filas),
                 "por_canal_referencia": por_canal(referencia), "filas": filas}
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos), encoding="utf-8")
        print(variante, nombre, f"{datos['minutos']:.1f} min", "exactitud", round(datos["resultado"]["exactitud"], 3),
              "auroc", round(datos["resultado"]["auroc"], 3), "| referencia",
              round(datos["referencia"]["exactitud"], 3), flush=True)
        salidas.append(datos)
    return salidas


def tabla(bloques):
    salida = {}
    for b in bloques:
        fila = salida.setdefault(b["variante"], {})
        fila[b["particion"]] = {k: round(b["resultado"][k], 3) for k in ("exactitud", "auroc", "puntaje_abierto")}
        fila.setdefault("_referencia", {})[b["particion"]] = round(b["referencia"]["exactitud"], 3)
    for fila in salida.values():
        valores = [v["exactitud"] for k, v in fila.items() if not k.startswith("_")]
        fila["_media_exactitud"] = round(float(np.mean(valores)), 4)
        fila["_media_auroc"] = round(float(np.mean([v["auroc"] for k, v in fila.items() if not k.startswith("_")])), 4)
    return salida


def modelo_final(ctx, variante, pasos):
    permitidas = np.ones(len(ctx.meta), bool)
    vistas, forma = armar_geometrias(ctx.meta, ctx.pool, permitidas)
    necesarias = ("imagen", "puntos") if variante == "multivista" else (variante,)
    modelos, valores = {}, {}
    for v in necesarias:
        modelos[v], valores[v] = entrenar(v, ctx.t, vistas, forma, pasos, 2026112499, ctx.disp)
    decisor = ajustar_decisor(modelos, variante, ctx.meta, ctx.pool, permitidas, 2026112499)
    DESTINO_FINAL.mkdir(parents=True, exist_ok=True)
    for v, m in modelos.items():
        guardar_descriptor(m.cpu(), DESTINO_FINAL / f"descriptor_{v}.pt", {"val": valores[v], "pasos": pasos,
                                                                          "tipo": v, "cadena": CADENA})
    if ctx.ruta_recuperador.resolve() != (DESTINO_FINAL / "recuperador.pt").resolve():
        shutil.copyfile(ctx.ruta_recuperador, DESTINO_FINAL / "recuperador.pt")
    guardar_paquete(DESTINO_FINAL / f"paquete_{variante}.joblib", decisor, CADENA, "recuperador.pt",
                    f"descriptor_{necesarias[0]}.pt",
                    {"entrenamiento": "canales_variados", "variante": variante, "vistas": len(ctx.meta),
                     "descriptores": {v: f"descriptor_{v}.pt" for v in necesarias}})
    print("modelo final guardado", DESTINO_FINAL, variante, flush=True)


# ──────────────────────────────  Entry  ──────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", choices=("cribado", "completo", "final"), default="cribado")
    p.add_argument("--variante", choices=VARIANTES)
    p.add_argument("--pasos", type=int)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Este entrenamiento necesita la RTX y CUDA no está disponible.")
    torch.set_num_threads(4)
    ctx = Contexto(torch.device("cuda"))
    planes = particiones()
    if a.etapa == "final":
        modelo_final(ctx, a.variante, a.pasos or 8000)
        return
    if a.etapa == "cribado":
        elegidas, variantes, pasos = CRIBADO, VARIANTES, a.pasos or 4000
    else:
        elegidas, variantes, pasos = tuple(planes), (a.variante,), a.pasos or 8000
    bloques = []
    for i, nombre in enumerate(elegidas):
        bloques += correr(ctx, variantes, nombre if a.etapa == "cribado" else f"{nombre}_completo", planes[nombre],
                          2026112420 + i, pasos)
    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen_total = {"etapa": a.etapa, "pasos": pasos, "tabla": tabla(bloques),
                     "por_canal": {b["variante"] + "/" + b["particion"]: b["por_canal"] for b in bloques}}
    (SALIDA / f"resumen_{a.etapa}.json").write_text(json.dumps(resumen_total, indent=1), encoding="utf-8")
    print(json.dumps(resumen_total["tabla"], indent=1), flush=True)


if __name__ == "__main__":
    main()
