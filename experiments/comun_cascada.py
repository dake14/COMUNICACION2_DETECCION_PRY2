"""Utilidades compartidas por los experimentos de la cascada."""
import json
from pathlib import Path

import numpy as np

from src.cascada import geometria as g
from src.cascada.decisor import DESCONOCIDA, INDETERMINADA
from src.cascada.sintesis import condicion_aleatoria, transmitir
from src.channels import add_awgn, apply_carrier_impairments, apply_multipath
from src.waveform import pulse_shape

CONFIG_E2 = Path("configs/dataset_e2.json")
DIR_MODELOS = Path("results/cascada/modelos")
RUTA_RECUPERADOR = DIR_MODELOS / "recuperador.pt"
RUTA_DESCRIPTOR = DIR_MODELOS / "descriptor.pt"
RUTA_PAQUETE = DIR_MODELOS / "paquete.joblib"


def canales_config(clave):
    datos = json.loads(CONFIG_E2.read_text(encoding="utf-8"))[clave]
    return [np.array([complex(a, b) for a, b in c]) for c in datos]


def transmitir_repo(puntos, n_simbolos, rng, snr_db, sps=8, beta=0.35, taps=None, cfo=0.0):
    """Cadena del proyecto original: src.waveform + src.channels."""
    total = n_simbolos + 24
    simbolos = np.asarray(puntos)[rng.integers(0, len(puntos), total)]
    x, _ = pulse_shape(simbolos, sps=sps, beta=beta, span=10)
    if taps is not None:
        x = apply_multipath(x, taps)
    x = x[10 * sps: 10 * sps + n_simbolos * sps]
    x = apply_carrier_impairments(x, 1.0, phase_rad=float(rng.uniform(-np.pi, np.pi)), cfo_hz=cfo)
    y, _ = add_awgn(x, snr_db, rng)
    return (y * rng.uniform(0.5, 1.5)).astype(np.complex64)


def conjunto_repo(semilla, nombres_puntos, por_caso, snrs=(0, 5, 10, 15, 20, 30), canales=None):
    canales = canales if canales is not None else [None, *canales_config("id_channels")[1:]]
    rng = np.random.default_rng(semilla)
    items = []
    for nombre, (puntos, etiqueta) in nombres_puntos.items():
        for snr in snrs:
            for _ in range(por_caso):
                sps = int(rng.choice([4, 8, 10]))
                taps = canales[int(rng.integers(len(canales)))]
                n = int(rng.choice([512, 1024]))
                cfo = float(rng.uniform(-0.0125, 0.0125)) * 8 / sps
                iq = transmitir_repo(puntos, n, rng, snr, sps, float(rng.choice([0.25, 0.35, 0.5])), taps, cfo)
                items.append({"iq": iq, "real": nombre, "etiqueta": etiqueta, "snr": snr, "sps": sps})
    return items


def conjunto_procedural(semilla, por_clase, desconocidas):
    """Catálogo y geometrías aleatorias con el generador de entrenamiento y tasa desconocida."""
    rng = np.random.default_rng(semilla)
    planes = [(k, v, k) for k, v in g.catalogo().items() for _ in range(por_clase)]
    planes += [("ALEATORIA", None, DESCONOCIDA)] * desconocidas
    items = []
    for real, puntos, etiqueta in planes:
        p = puntos if puntos is not None else g.geometria_aleatoria(rng)
        sps = float(rng.choice([4.0, 8.0, 10.0, 5.3, 6.0, 12.0]))
        cond = condicion_aleatoria(rng, snr_rango=(0.0, 30.0), cfo_max=0.008, prob_canal=0.6)
        iq = transmitir(p, int(rng.choice([512, 1024, 2048])), cond, rng, sps_salida=sps)["iq"]
        items.append({"iq": iq, "real": real, "etiqueta": etiqueta, "snr": cond.snr_db, "sps": sps})
    return items


def grupo_validacion():
    d = {k: (v, k) for k, v in g.catalogo().items()}
    d.update({k: (v, DESCONOCIDA) for k, v in g.desconocidas_validacion().items()})
    return d


def metricas(filas, nombres_catalogo):
    """Métricas por archivo. Cada fila: etiqueta, prediccion, confianza, probabilidad_fondo."""
    salida = {"n": len(filas)}
    conocidas = [f for f in filas if f["etiqueta"] in nombres_catalogo]
    desconocidas = [f for f in filas if f["etiqueta"] == DESCONOCIDA]
    if conocidas:
        salida["exactitud_cerrada"] = float(np.mean([f["cerrada"] == f["etiqueta"] for f in conocidas]))
        salida["exactitud_abierta"] = float(np.mean([f["prediccion"] == f["etiqueta"] for f in conocidas]))
        salida["cobertura_conocidas"] = float(np.mean([f["prediccion"] != INDETERMINADA for f in conocidas]))
        salida["por_clase_cerrada"] = {
            c: float(np.mean([f["cerrada"] == c for f in conocidas if f["etiqueta"] == c]))
            for c in nombres_catalogo if any(f["etiqueta"] == c for f in conocidas)}
        salida["f1_macro_cerrada"] = _f1(conocidas, nombres_catalogo)
    if desconocidas:
        salida["rechazo_desconocidas"] = float(np.mean([f["prediccion"] == DESCONOCIDA for f in desconocidas]))
        salida["falsa_aceptacion"] = float(np.mean([f["prediccion"] in nombres_catalogo for f in desconocidas]))
    if conocidas and desconocidas:
        y = np.array([0] * len(conocidas) + [1] * len(desconocidas))
        s = np.array([f["p_desconocida"] for f in conocidas + desconocidas])
        salida["auroc_desconocidas"] = auroc(y, s)
    decididas = [f for f in filas if f["confianza"] is not None]
    if decididas:
        salida["ece"] = ece([f["confianza"] for f in decididas],
                            [f["top"] == f["etiqueta"] for f in decididas])
    return salida


def _f1(filas, clases):
    valores = []
    for c in clases:
        tp = sum(f["cerrada"] == c and f["etiqueta"] == c for f in filas)
        fp = sum(f["cerrada"] == c and f["etiqueta"] != c for f in filas)
        fn = sum(f["cerrada"] != c and f["etiqueta"] == c for f in filas)
        valores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(valores))


def auroc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def ece(confianza, acierto, bins=10):
    c, a = np.asarray(confianza, float), np.asarray(acierto, float)
    total = 0.0
    for i in range(bins):
        m = (c > i / bins) & (c <= (i + 1) / bins)
        if m.any():
            total += m.mean() * abs(c[m].mean() - a[m].mean())
    return float(total)


def fila_resultado(item, r, nombres_catalogo):
    p = r.get("probabilidades") or {}
    cerrada = max(nombres_catalogo, key=lambda n: p.get(n, -1.0)) if p else None
    top = max(p, key=p.get) if p else None
    return {"real": item["real"], "etiqueta": item["etiqueta"], "snr": item["snr"], "sps": item.get("sps"),
            "prediccion": r["modulacion"], "cerrada": cerrada, "top": top, "confianza": r["confianza"],
            "p_desconocida": p.get(DESCONOCIDA, 1.0), "sps_est": (r.get("etapas") or {}).get("sps_origen"),
            "puntos": (r.get("descripcion") or {}).get("puntos_aprox"),
            "anillos": (r.get("descripcion") or {}).get("anillos_aprox")}
