# ─────────────────────────────  Imports  ─────────────────────────────
import numpy as np
from scipy.interpolate import CubicSpline

from src.cascada.clasica import SPS, fase_muestreo

# ─────────────────────────────  Config  ──────────────────────────────
ANGULOS = np.linspace(0, np.pi, 91, endpoint=False)
LIMITE = 2.2
CASILLAS = 65
MARGEN_SIMBOLOS = 16
PUNTOS_TRAZA = 33


# ───────────────────────────────  Core  ──────────────────────────────
def unitaria(z):
    return z / np.sqrt(max(float(np.mean(np.abs(z) ** 2)), 1e-12))


def _entropia(valores):
    cuenta, _ = np.histogram(valores, bins=CASILLAS, range=(-LIMITE, LIMITE))
    p = cuenta[cuenta > 0] / max(cuenta.sum(), 1)
    return float(-np.sum(p * np.log(p)))


def _entropia_ejes(z):
    return _entropia(z.real) + _entropia(z.imag)


def angulo_de_ejes(z):
    angulo = min(ANGULOS, key=lambda a: _entropia_ejes(z * np.exp(-1j * a)))
    girada = z * np.exp(-1j * angulo)
    return angulo + np.pi / 2 if np.mean(girada.imag ** 2) > np.mean(girada.real ** 2) else angulo


def alinear(z):
    z = unitaria(np.asarray(z, dtype=np.complex128))
    return z * np.exp(-1j * angulo_de_ejes(z))


def muestra_uniforme(z, maximo):
    return z[np.linspace(0, len(z) - 1, min(maximo, len(z))).astype(int)] if len(z) else z


def _rejilla_t2(onda):
    tau = fase_muestreo(onda, SPS)
    rejilla = np.arange(tau[0] % SPS, len(onda) - SPS, SPS / 2)
    desvio = np.interp(rejilla, np.arange(len(tau)), tau - tau[0])
    return np.clip(rejilla + desvio, 0, len(onda) - 1)


def _elegir_simbolos(total, maximo):
    margen = min(MARGEN_SIMBOLOS, total // 8)
    disponibles = np.arange(max(margen, 2), total - max(margen, 2))
    return muestra_uniforme(disponibles, maximo)


def _salida_ecualizador(r, rejilla, curva, indices, desplazamiento):
    pesos = np.asarray(r["pesos"])
    tomas = np.arange(len(pesos))
    u = 2.0 * indices[:, None, None] + tomas[None, None, :] + 2.0 * desplazamiento[None, :, None]
    x = curva(np.interp(u, np.arange(len(rejilla)), rejilla)) * np.exp(-1j * np.pi * r["df_base"] * u)
    y = x @ pesos
    return y * np.exp(-1j * np.pi * r["ajuste"] * (2.0 * indices[:, None] + 2.0 * desplazamiento[None, :]))


def trazas_ojo(r, maximo=160):
    onda = np.asarray(r["onda"], dtype=np.complex128)
    simbolos = alinear(r["simbolos"])
    indices = _elegir_simbolos(len(simbolos), maximo)
    if len(indices) < 8:
        return {"t": [], "i": [], "q": []}
    t = np.linspace(-1.0, 1.0, PUNTOS_TRAZA)
    centro = PUNTOS_TRAZA // 2
    curva = CubicSpline(np.arange(len(onda), dtype=float), onda)
    trazas = _salida_ecualizador(r, _rejilla_t2(onda), curva, indices, t)
    referencia = trazas[:, centro]
    ganancia = np.vdot(referencia, simbolos[indices]) / max(float(np.vdot(referencia, referencia).real), 1e-12)
    trazas = trazas * ganancia
    return {"t": redondear(t), "i": redondear(trazas.real), "q": redondear(trazas.imag)}


def redondear(valores, decimales=3):
    return np.round(np.asarray(valores, dtype=float), decimales).tolist()


def puntos_xy(z):
    return {"x": redondear(z.real), "y": redondear(z.imag)}
