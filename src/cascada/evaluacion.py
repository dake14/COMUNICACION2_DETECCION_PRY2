"""Medidas de calidad de la recuperación."""
import numpy as np


def evm_ciego(z, puntos, giros=180):
    """EVM al punto más cercano, con la mejor fase y escala global."""
    z = np.asarray(z, dtype=np.complex128)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2))
    puntos = np.asarray(puntos)
    muestra = z[:: max(1, len(z) // 1500)]
    mejor, mejor_giro = np.inf, 1.0
    for a in np.exp(-2j * np.pi * np.arange(giros) / giros):
        d = np.abs((muestra * a)[:, None] - puntos[None, :]).min(axis=1)
        e = float(np.mean(d ** 2))
        if e < mejor:
            mejor, mejor_giro = e, a
    rotado = z * mejor_giro
    cercano = puntos[np.abs(rotado[:, None] - puntos[None, :]).argmin(axis=1)]
    escala = np.vdot(rotado, cercano) / np.vdot(rotado, rotado)
    return float(np.sqrt(np.mean(np.abs(escala * rotado - cercano) ** 2)))


def evm_referencia(z, simbolos, max_retardo=40):
    """EVM contra los símbolos transmitidos, alineando retardo y ganancia por tramos."""
    z = np.asarray(z, dtype=np.complex128)
    s = np.asarray(simbolos, dtype=np.complex128)
    n = min(len(z), len(s))
    if n < 2 * max_retardo + 16:
        return float("nan")
    mejor = (0.0, 0)
    for d in range(-max_retardo, max_retardo + 1):
        a = z[max(d, 0): n + min(d, 0)]
        b = s[max(-d, 0): n - max(d, 0)]
        m = min(len(a), len(b))
        tramos = range(0, m - 31, 32)
        c = sum(abs(np.vdot(a[i:i + 32], b[i:i + 32])) for i in tramos)
        if c > mejor[0]:
            mejor = (c, d)
    d = mejor[1]
    a = z[max(d, 0): n + min(d, 0)]
    b = s[max(-d, 0): n - max(d, 0)]
    m = min(len(a), len(b))
    error, energia = 0.0, 0.0
    for i in range(0, m - 31, 32):
        u, v = a[i:i + 32], b[i:i + 32]
        g = np.vdot(u, v) / max(np.vdot(u, u).real, 1e-12)
        error += float(np.sum(np.abs(g * u - v) ** 2))
        energia += float(np.sum(np.abs(v) ** 2))
    return float(np.sqrt(error / max(energia, 1e-12)))
