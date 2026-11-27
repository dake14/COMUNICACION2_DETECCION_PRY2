# ─────────────────────────────  Imports  ─────────────────────────────
import numpy as np

from src.cascada.autofoco import autofoco, entropia
from src.cascada.cadena import DF_FINO, MAX_AJUSTE, MAX_ENTROPIA, MAX_MUESTRAS, MIN_SIMBOLOS
from src.cascada.clasica import (LARGO_FSE, filtro_adaptado, peso_inicial, pesos_cma, refinar_ciego, simbolos_t2,
                                 ventanas_t2)
from src.cascada.normalizador import normalizar
from src.cascada.recuperador import aplicar_recuperador

# ─────────────────────────────  Config  ──────────────────────────────
RUTAS = {"mixta": ("red", "clasica", "directa")}


# ───────────────────────────────  Core  ──────────────────────────────
def _enfocar(iq8, ruta, recuperador):
    iq8 = np.asarray(iq8)[:MAX_MUESTRAS]
    onda = aplicar_recuperador(recuperador, iq8) if ruta == "red" else filtro_adaptado(iq8)
    x2 = simbolos_t2(onda)
    pesos = pesos_cma(x2[:MAX_AJUSTE]) if ruta == "clasica" else peso_inicial()
    if len(x2) < 2 * MIN_SIMBOLOS + LARGO_FSE:
        return None
    _, df, nitidez = autofoco(ventanas_t2(x2[:MAX_AJUSTE]) @ pesos)
    x2 = x2 * np.exp(-1j * np.pi * df * np.arange(len(x2)))
    pesos = refinar_ciego(x2[:MAX_AJUSTE], pesos, lambda y: entropia(y[:MAX_ENTROPIA]))
    simbolos, ajuste, _ = autofoco(ventanas_t2(x2) @ pesos, df_max=DF_FINO)
    return {"simbolos": simbolos, "df": df + ajuste, "df_base": df, "ajuste": ajuste, "nitidez": nitidez,
            "pesos": pesos, "onda": onda, "ruta": ruta}


def _mejor_ruta(iq8, cadena, recuperador):
    mejor, menor = None, np.inf
    for ruta in RUTAS.get(cadena, (cadena,)):
        salida = _enfocar(iq8, ruta, recuperador)
        if salida is None:
            continue
        valor = entropia(salida["simbolos"][:MAX_ENTROPIA])
        if valor < menor:
            mejor, menor = salida, valor
    return mejor


def recuperar_detalle(iq, cadena, recuperador=None, sps=None):
    for hipotesis in sorted(normalizar(iq, sps=sps), key=lambda h: -h.puntaje):
        if len(hipotesis.iq) < MIN_SIMBOLOS * 8:
            continue
        salida = _mejor_ruta(hipotesis.iq, cadena, recuperador)
        if salida is not None:
            return {**salida, "sps_origen": hipotesis.sps_origen, "cfo_grueso": hipotesis.cfo_grueso}
    return None
