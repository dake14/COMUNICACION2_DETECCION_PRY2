"""Une normalización, recuperación y autofoco hasta obtener símbolos."""
import numpy as np

from src.cascada.autofoco import autofoco, entropia
from src.cascada.clasica import (LARGO_FSE, filtro_adaptado, pesos_cma, peso_inicial, refinar_ciego,
                                 simbolos_t2, ventanas_t2)
from src.cascada.normalizador import normalizar
from src.cascada.recuperador import aplicar_recuperador
from src.cascada.seguimiento import cma_seguimiento, fase_por_bloques

MIN_SIMBOLOS = 24
MAX_MUESTRAS = 32768
MAX_ENTROPIA = 1024
MAX_AJUSTE = 3072
DF_FINO = 0.004


def _inicio(iq8, ruta, recuperador):
    """Señal a dos muestras por símbolo y pesos iniciales de cada ruta."""
    iq8 = np.asarray(iq8)[:MAX_MUESTRAS]
    if ruta == "red":
        return simbolos_t2(aplicar_recuperador(recuperador, iq8)), peso_inicial()
    x2 = simbolos_t2(filtro_adaptado(iq8))
    return x2, pesos_cma(x2[:MAX_AJUSTE]) if ruta == "clasica" else peso_inicial()


def _enfocar_ruta(iq8, ruta, recuperador, refinar):
    seguir = ruta == "seguimiento"
    x2, w = _inicio(iq8, "red" if seguir else ruta, recuperador)
    if len(x2) < 2 * MIN_SIMBOLOS + LARGO_FSE:
        return None
    _, df, nitidez = autofoco(ventanas_t2(x2[:MAX_AJUSTE]) @ w)
    x2 = x2 * np.exp(-1j * np.pi * df * np.arange(len(x2)))
    if refinar:
        w = refinar_ciego(x2[:MAX_AJUSTE], w, lambda y: entropia(y[:MAX_ENTROPIA]))
    y = cma_seguimiento(x2, w)[0] if seguir else ventanas_t2(x2) @ w
    z, ajuste, _ = autofoco(y, df_max=DF_FINO)
    if seguir:
        z = fase_por_bloques(z)
    return z, df + ajuste, nitidez


RUTAS = {"mixta": ("red", "clasica", "directa"), "mixta_seguimiento": ("red", "clasica", "directa", "seguimiento")}


def enfocar(iq8, cadena="red", recuperador=None, refinar=True):
    """Devuelve (símbolos, df, nitidez, ruta). `mixta` elige la salida de menor entropía."""
    rutas = RUTAS.get(cadena, (cadena,))
    mejor = None
    for ruta in rutas:
        salida = _enfocar_ruta(iq8, ruta, recuperador, refinar)
        if salida is None:
            continue
        z, df, nitidez = salida
        h = entropia(z[:MAX_ENTROPIA])
        if mejor is None or h < mejor[0]:
            mejor = (h, z, df, nitidez, ruta)
    return None if mejor is None else mejor[1:]


def recuperar(iq, cadena="red", recuperador=None, sps=None):
    """Usa la hipótesis de tasa con la línea cíclica más fuerte que dé símbolos suficientes."""
    for h in sorted(normalizar(iq, sps=sps), key=lambda h: -h.puntaje):
        if len(h.iq) < MIN_SIMBOLOS * 8:
            continue
        salida = enfocar(h.iq, cadena, recuperador)
        if salida is None:
            continue
        z, df, nitidez, ruta = salida
        return {"simbolos": z, "df": df, "nitidez": nitidez, "sps_origen": h.sps_origen,
                "puntaje_tasa": h.puntaje, "cfo_grueso": h.cfo_grueso, "ruta": ruta}
    return None
