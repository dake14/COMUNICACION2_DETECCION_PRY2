# ─────────────────────────────  Imports  ─────────────────────────────
import numpy as np
import torch

from src.cascada.pipeline import calidad_de, huella_de_simbolos
from src.familias.ajuste import medir
from src.familias.conforme import salida_final
from src.familias.detalle import vistas_constelacion
from src.familias.especialistas import log_probabilidades
from src.familias.integracion import fusionar
from src.familias.tasa import estimar_tasa
from src.familias.tonos import analizar_fsk


# ─────────────────────────────  Config  ──────────────────────────────
MAX_SIMBOLOS_ESPECIALISTA = 2048


# ───────────────────────────────  Core  ──────────────────────────────
def _p_decisor(r, snr, paquete):
    if r is None:
        return None, None
    z, geometria = huella_de_simbolos(r["simbolos"], paquete)
    decisor = paquete["decisor"]
    return dict(zip(decisor.nombres, decisor.probabilidades(z, calidad_de(r, snr))[0].tolist())), (z, geometria)


def _p_especialista(r, paquete):
    if r is None or "especialista_modelo" not in paquete:
        return None
    modelo, clases = paquete["especialista_modelo"]
    vista = vistas_constelacion(np.asarray(r["simbolos"])[:MAX_SIMBOLOS_ESPECIALISTA])[None]
    return dict(zip(clases, np.exp(log_probabilidades(modelo, vista)[0]).tolist()))


def _consistentes(r, paquete):
    if r is None or "consistencia" not in paquete:
        return None
    medicion = medir(r["simbolos"], paquete["plantillas"])
    peso = float(paquete["peso_inconsistente"])
    return {c: (1.0 if ok else peso) for c, ok in paquete["consistencia"].consistentes(medicion).items()}


def _no_caben(previas, consistentes, n=3):
    if not consistentes:
        return []
    candidatas = sorted((c for c in previas if c in consistentes), key=lambda c: -previas[c])[:n]
    return [c for c in candidatas if consistentes[c] < 1.0 and previas[c] >= 0.05]


def _tonos(x):
    tasa = estimar_tasa(x)
    if not tasa["sps"]:
        return None
    t = analizar_fsk(x, tasa["sps"])
    return {"tonos": int(t["tonos"]), "h": round(float(t["h"]), 3), "sps": round(float(tasa["sps"]), 2),
            "puntaje": float(t["puntaje"])}


def _enrutador(analisis, rama, tonos, fiable, p_especialista=None, grupos=()):
    familias = sorted(({"nombre": f, "valor": round(v, 4)} for f, v in analisis["familias"].items()),
                      key=lambda f: -f["valor"])
    especialistas = []
    for vista, votos in analisis["especialistas"].items():
        clase = max(votos, key=votos.get)
        especialistas.append({"vista": vista, "clase": clase, "valor": round(votos[clase], 4)})
    if p_especialista and max(analisis["familias"], key=analisis["familias"].get) in {f for g in grupos for f in g}:
        clase = max(p_especialista, key=p_especialista.get)
        especialistas.append({"vista": "constelacion", "clase": clase, "valor": round(p_especialista[clase], 4)})
    visibles = None if tonos is None or rama != "frecuencia" else {**{k: tonos[k] for k in ("tonos", "h", "sps")},
                                                                   "fiable": fiable}
    return {"familias": familias, "especialistas": especialistas, "rama": rama, "tonos": visibles}


def _decision(probabilidades, umbral, paquete):
    orden = paquete.get("orden") or list(probabilidades)
    p = np.array([probabilidades[n] for n in orden])
    conjunto = paquete["conforme"].conjunto(p)[0] if "conforme" in paquete else [int(p.argmax())]
    s = salida_final(p, orden, conjunto, paquete.get("equivalentes", {}), umbral)
    return {"modulacion": s["modulacion"], "confianza": round(s["confianza"], 4), "mas_probable": s["mas_probable"],
            "misma_senal_que": s["misma_senal_que"],
            "tambien_puede_ser": [{"nombre": a["nombre"], "probabilidad": round(a["probabilidad"], 4)}
                                  for a in s["tambien_puede_ser"]]}


def salida_integrada(x, r, snr, paquete):
    with torch.no_grad():
        analisis = paquete["enrutador_modelo"].analizar(x)
    p_decisor, huella = _p_decisor(r, snr, paquete)
    tonos = _tonos(x) if paquete.get("con_tonos", True) else None
    fiable = tonos is not None and tonos["puntaje"] >= paquete.get("puntaje_tonos", 0.8)
    p_especialista = _p_especialista(r, paquete)
    argumentos = (analisis["familias"], analisis["clases"], p_decisor, paquete["familia_de"],
                  tonos["tonos"] if fiable else None)
    opciones = {"grupos": paquete.get("grupos") if p_especialista else None, "p_especialista": p_especialista}
    consistentes = _consistentes(r, paquete)
    fusion = fusionar(*argumentos, **opciones, consistentes=consistentes)
    previas = fusionar(*argumentos, **opciones)["probabilidades"] if consistentes else {}
    umbral = float(paquete["decisor"].umbral)
    decision = _decision(fusion["probabilidades"], umbral, paquete)
    probabilidades = sorted(({"nombre": n, "valor": round(float(v), 4)} for n, v in fusion["probabilidades"].items()),
                            key=lambda p: -p["valor"])
    return {"decision": {**decision, "umbral": round(umbral, 3), "no_cabe_en": _no_caben(previas, consistentes)},
            "probabilidades": probabilidades,
            "enrutador": _enrutador(analisis, fusion["rama"], tonos, fiable, p_especialista,
                                    paquete.get("grupos") or ())}, huella
