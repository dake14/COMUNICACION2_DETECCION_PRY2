"""Une normalización, recuperación y autofoco hasta obtener símbolos."""
import numpy as np

from src.cascada.autofoco import autofoco
from src.cascada.clasica import simbolos_clasicos, simbolos_sin_ecualizar, tomar_simbolos
from src.cascada.normalizador import normalizar
from src.cascada.recuperador import aplicar_recuperador

MIN_SIMBOLOS = 24
MAX_MUESTRAS = 32768


def simbolos_de(iq8, cadena="red", recuperador=None):
    iq8 = np.asarray(iq8)[:MAX_MUESTRAS]
    if cadena == "red":
        return tomar_simbolos(aplicar_recuperador(recuperador, iq8))[4:-4]
    if cadena == "clasica":
        return simbolos_clasicos(iq8)[8:]
    if cadena == "directa":
        return simbolos_sin_ecualizar(iq8)[8:-8]
    raise ValueError(f"Cadena desconocida: {cadena}")


def recuperar(iq, cadena="red", recuperador=None, sps=None):
    """Prueba cada hipótesis de tasa y conserva la constelación más nítida."""
    mejor = None
    for h in normalizar(iq, sps=sps):
        if len(h.iq) < MIN_SIMBOLOS * 8:
            continue
        crudos = simbolos_de(h.iq, cadena, recuperador)
        if len(crudos) < MIN_SIMBOLOS:
            continue
        z, df, nitidez = autofoco(crudos)
        if mejor is None or nitidez > mejor["nitidez"]:
            mejor = {"simbolos": z, "df": df, "nitidez": nitidez, "sps_origen": h.sps_origen,
                     "puntaje_tasa": h.puntaje, "cfo_grueso": h.cfo_grueso}
    return mejor
