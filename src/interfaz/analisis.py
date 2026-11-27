# ─────────────────────────────  Imports  ─────────────────────────────
import time

import numpy as np

from src.cascada.decisor import INDETERMINADA
from src.cascada.descriptor import imagen_densidad
from src.cascada.detector import detectar_ciego, snr_ciega_db
from src.cascada.pipeline import MIN_MUESTRAS, calidad_de, huella_de_simbolos
from src.interfaz.graficas import alinear, muestra_uniforme, puntos_xy, redondear, trazas_ojo, unitaria
from src.interfaz.recuperacion import recuperar_detalle

# ─────────────────────────────  Config  ──────────────────────────────
LIMITE_MUESTRAS = 200_000
PUNTOS_CONSTELACION = 4000
PUNTOS_ENTRADA = 3000


# ───────────────────────────────  Core  ──────────────────────────────
def _ventana(iq, inicio):
    x = np.asarray(iq, dtype=np.complex128).ravel()
    if not 0 <= inicio < max(len(x), 1):
        raise ValueError(f"La muestra inicial debe estar entre 0 y {max(len(x) - 1, 0)}.")
    x = x[inicio:inicio + LIMITE_MUESTRAS]
    if len(x) < MIN_MUESTRAS:
        raise ValueError(f"Se necesitan al menos {MIN_MUESTRAS} muestras y la ventana tiene {len(x)}.")
    if not np.isfinite(x).all() or np.mean(np.abs(x - x.mean()) ** 2) <= 0:
        raise ValueError("La señal no tiene potencia útil o contiene valores inválidos.")
    return x


def _imagenes_cnn(simbolos):
    imagen = imagen_densidad(simbolos)
    maximos = np.maximum(imagen.max(axis=(1, 2), keepdims=True), 1e-9)
    imagen = (imagen / maximos * 255).astype(np.uint8)
    return {"cartesiana": imagen[0].tolist(), "polar": imagen[1].tolist()}


def _salida_cnn(r, snr, paquete):
    z, geometria = huella_de_simbolos(r["simbolos"], paquete)
    decision = paquete["decisor"].decidir(z, calidad_de(r, snr))
    probabilidades = sorted(({"nombre": n, "valor": round(float(v), 4)} for n, v in decision["probabilidades"].items()),
                            key=lambda p: -p["valor"])
    return {"decision": {"modulacion": decision["modulacion"], "confianza": round(decision["confianza"], 4),
                         "umbral": round(float(paquete["decisor"].umbral), 3)},
            "probabilidades": probabilidades,
            "cnn": {"entrada": _imagenes_cnn(r["simbolos"]), "huella": redondear(z[0], 4),
                    "puntos_aprox": int(round(2 ** float(geometria[0, 0] * 7))),
                    "anillos_aprox": max(1, int(round(float(geometria[0, 1] * 6))))}}


def _etapas(r, snr, deteccion, milisegundos):
    return {"sps": round(float(r["sps_origen"]), 3), "cfo": round(float(r["cfo_grueso"] + r["df"] / r["sps_origen"]), 6),
            "ruta": r["ruta"], "simbolos": len(r["simbolos"]), "snr_db": None if snr is None else round(float(snr), 1),
            "detectada": None if deteccion is None else bool(deteccion.get("detectada")),
            "nitidez": round(float(r["nitidez"]), 2), "tiempo_ms": round(milisegundos)}


def analizar_senal(iq, paquete, sps=None, inicio=0):
    comienzo = time.perf_counter()
    x = _ventana(iq, inicio)
    deteccion, snr = detectar_ciego(x), snr_ciega_db(x)
    entrada = puntos_xy(muestra_uniforme(unitaria(x - x.mean()), PUNTOS_ENTRADA))
    base = {"muestras_totales": int(np.size(iq)), "muestras_analizadas": len(x), "inicio": inicio, "entrada": entrada}
    r = recuperar_detalle(x, paquete["cadena"], paquete.get("recuperador"), sps)
    if r is None:
        return {**base, "decision": {"modulacion": INDETERMINADA, "confianza": None, "umbral": None},
                "aviso": "No se pudo identificar una tasa de símbolo en esta señal."}
    salida = _salida_cnn(r, snr, paquete)
    constelacion = puntos_xy(muestra_uniforme(alinear(r["simbolos"]), PUNTOS_CONSTELACION))
    ojo = trazas_ojo(r)
    milisegundos = 1000 * (time.perf_counter() - comienzo)
    return {**base, **salida, "constelacion": constelacion, "ojo": ojo,
            "etapas": _etapas(r, snr, deteccion, milisegundos)}
