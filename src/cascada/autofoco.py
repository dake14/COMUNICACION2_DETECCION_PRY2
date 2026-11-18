"""Etapa 3: elimina el giro residual minimizando la entropía de la constelación."""
import numpy as np

BINS = 48
LIMITE = 2.4
DF_MAX = 0.04
MAX_SIMBOLOS = 1024


def normalizar_potencia(z):
    z = np.asarray(z, dtype=np.complex128)
    return z / np.sqrt(max(np.mean(np.abs(z) ** 2), 1e-12))


def entropia(z, bins=BINS):
    ejes = np.clip(((np.stack((z.real, z.imag)) + LIMITE) / (2 * LIMITE) * bins).astype(int), 0, bins - 1)
    cuenta = np.bincount(ejes[0] * bins + ejes[1], minlength=bins * bins)
    p = cuenta[cuenta > 0] / len(z)
    return float(-np.sum(p * np.log(p)))


def _buscar(z, n, centro, ancho, pasos):
    candidatos = centro + np.linspace(-ancho, ancho, pasos)
    valores = [entropia(z * np.exp(-2j * np.pi * df * n)) for df in candidatos]
    k = int(np.argmin(valores))
    return float(candidatos[k]), float(valores[k]), np.asarray(valores)


def autofoco(simbolos, df_max=DF_MAX):
    """Devuelve (símbolos corregidos, df en ciclos/símbolo, nitidez >= 0)."""
    z = normalizar_potencia(simbolos)
    muestra = z[: MAX_SIMBOLOS]
    n = np.arange(len(muestra))
    resolucion = 1.0 / (6 * max(len(muestra), 16))
    pasos = int(min(1201, 2 * df_max / resolucion + 1))
    df, _, valores = _buscar(muestra, n, 0.0, df_max, pasos)
    paso = 2 * df_max / (pasos - 1)
    for _ in range(2):
        df, mejor, _ = _buscar(muestra, n, df, paso, 9)
        paso /= 4
    corregidos = z * np.exp(-2j * np.pi * df * np.arange(len(z)))
    nitidez = float((np.median(valores) - mejor) / max(np.std(valores), 1e-9))
    return corregidos, df, max(nitidez, 0.0)
