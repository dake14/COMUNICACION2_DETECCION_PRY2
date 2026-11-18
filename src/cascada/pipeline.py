"""Tubería completa de la cascada sobre un archivo I/Q."""
from pathlib import Path

import joblib
import numpy as np

from src.cascada.cadena import recuperar
from src.cascada.decisor import INDETERMINADA
from src.cascada.descriptor import calcular_huellas, cargar_descriptor, imagen_densidad
from src.cascada.detector import detectar_ciego, snr_ciega_db
from src.cascada.recuperador import cargar_recuperador

PAQUETE_DEFECTO = Path("results/cascada/modelos/paquete.joblib")
MIN_MUESTRAS = 256


def cargar_paquete(ruta=PAQUETE_DEFECTO):
    ruta = Path(ruta)
    datos = joblib.load(ruta)
    base = ruta.parent
    datos["recuperador"] = cargar_recuperador(base / datos["ruta_recuperador"]) if datos.get("ruta_recuperador") else None
    datos["descriptor"] = cargar_descriptor(base / datos["ruta_descriptor"])
    return datos


def guardar_paquete(ruta, decisor, cadena, ruta_recuperador, ruta_descriptor, metadatos=None):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"decisor": decisor, "cadena": cadena, "ruta_recuperador": ruta_recuperador,
                 "ruta_descriptor": ruta_descriptor, "metadatos": metadatos or {}}, ruta, compress=3)


def _indeterminada(estado, deteccion=None, snr=None):
    return {"modulacion": INDETERMINADA, "confianza": None, "probabilidades": {}, "estado": estado,
            "deteccion": deteccion, "snr_ciega_db": snr, "descripcion": None, "etapas": {}, "huella": None}


def huella_de_simbolos(simbolos, paquete):
    return calcular_huellas(paquete["descriptor"], imagen_densidad(simbolos)[None])


def analizar_cascada(iq, paquete, sps=None):
    x = np.asarray(iq, dtype=np.complex128).ravel()
    if len(x) < MIN_MUESTRAS:
        return _indeterminada("senal_muy_corta")
    if not np.isfinite(x).all() or np.mean(np.abs(x - x.mean()) ** 2) <= 0:
        return _indeterminada("senal_sin_potencia_util")
    deteccion = detectar_ciego(x)
    snr = snr_ciega_db(x)
    r = recuperar(x, paquete["cadena"], paquete.get("recuperador"), sps=sps)
    if r is None:
        return _indeterminada("tasa_no_identificada", deteccion, snr)
    z, g = huella_de_simbolos(r["simbolos"], paquete)
    decision = paquete["decisor"].decidir(z)
    return {**decision, "estado": "clasificada", "deteccion": deteccion, "snr_ciega_db": snr,
            "descripcion": {"puntos_aprox": int(round(2 ** float(g[0, 0] * 7))),
                            "anillos_aprox": max(1, int(round(float(g[0, 1] * 6))))},
            "etapas": {"sps_origen": r["sps_origen"], "cfo_grueso": r["cfo_grueso"],
                       "df_autofoco": r["df"], "nitidez": r["nitidez"], "simbolos": len(r["simbolos"])},
            "huella": z[0]}


def clasificacion_monitor(iq, detectada, paquete, sps=None):
    """Convierte el resultado al contrato `classification` del monitor."""
    salida = {"modulation": "NO_SIGNAL", "status": "no_signal_detected", "confidence": None,
              "confidence_calibrated": False, "confidence_status": "not_applicable_no_signal",
              "motor": "cascada"}
    if detectada is False:
        return salida
    r = analizar_cascada(iq, paquete, sps=sps)
    salida.update(modulation=r["modulacion"], status=r["estado"], confidence=r["confianza"],
                  confidence_calibrated=r["confianza"] is not None,
                  confidence_status="posterior_calibrada_en_validacion" if r["confianza"] is not None else "no_disponible",
                  probabilidades=r["probabilidades"], descripcion=r["descripcion"], etapas=r["etapas"],
                  snr_ciega_db=r["snr_ciega_db"])
    return salida
