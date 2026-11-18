"""Constelaciones nombradas y geometrías procedurales sin etiquetas."""
import numpy as np

SEMILLA_ARB_A = 7310021
SEMILLA_ARB_B = 7310022
ANGULOS = np.exp(-2j * np.pi * np.arange(72) / 72)


def normalizar_puntos(puntos):
    z = np.unique(np.round(np.asarray(puntos, dtype=np.complex128), 10))
    z = z - z.mean()
    return z / np.sqrt(np.mean(np.abs(z) ** 2))


def psk(m, giro=0.0):
    return normalizar_puntos(np.exp(1j * (2 * np.pi * np.arange(m) / m + giro)))


def rejilla(a, b=None):
    b = a if b is None else b
    x = np.arange(a) - (a - 1) / 2
    y = np.arange(b) - (b - 1) / 2
    return (x[:, None] + 1j * y[None, :]).ravel()


def qam(m):
    lado = int(round(np.sqrt(m)))
    return normalizar_puntos(rejilla(lado))


def qam_cruz(lado, esquina):
    z = rejilla(lado)
    limite = (lado - 1) / 2 - esquina
    return normalizar_puntos(z[~((np.abs(z.real) > limite) & (np.abs(z.imag) > limite))])


def pam(m):
    return normalizar_puntos(np.arange(m) - (m - 1) / 2 + 0j)


def apsk(cuentas, radios, giros=None):
    giros = giros if giros is not None else [0.0] * len(cuentas)
    partes = [r * np.exp(1j * (2 * np.pi * np.arange(c) / c + g))
              for c, r, g in zip(cuentas, radios, giros)]
    return normalizar_puntos(np.concatenate(partes))


def hexagonal(radio):
    puntos = []
    for q in range(-radio, radio + 1):
        for r in range(max(-radio, -q - radio), min(radio, -q + radio) + 1):
            puntos.append(q + r * np.exp(1j * np.pi / 3))
    return normalizar_puntos(puntos)


def catalogo():
    return {
        "BPSK": psk(2),
        "QPSK": psk(4, np.pi / 4),
        "8PSK": psk(8),
        "16QAM": qam(16),
        "64QAM": qam(64),
        "4PAM": pam(4),
    }


def desconocidas_validacion():
    return {
        "6PSK": psk(6),
        "24QAM": _qam24(),
        "HEX19": hexagonal(2),
        "ANILLOS12": apsk([2, 4, 6], [0.5, 1.2, 2.0], [0.0, 0.4, 0.1]),
    }


def _qam24():
    z = rejilla(6)
    return normalizar_puntos(z[np.abs(z.real) + np.abs(z.imag) < 3.5])


def _arbitraria(semilla, m):
    return _puntos_disco(np.random.default_rng(semilla), m)


def desconocidas_prueba():
    return {
        "32QAM": qam_cruz(6, 1),
        "128QAM": qam_cruz(12, 2),
        "16APSK": apsk([4, 12], [1.0, 2.57], [np.pi / 4, np.pi / 12]),
        "32APSK": apsk([4, 12, 16], [1.0, 2.84, 5.27], [np.pi / 4, np.pi / 12, 0.0]),
        "12PSK": psk(12),
        "8QAM_ESTRELLA": apsk([4, 4], [1.0, 1.0 + np.sqrt(3)], [0.0, np.pi / 4]),
        "ARB_A": _arbitraria(SEMILLA_ARB_A, 11),
        "ARB_B": _arbitraria(SEMILLA_ARB_B, 23),
    }


def distancia_minima(puntos):
    d = np.abs(puntos[:, None] - puntos[None, :])
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def distancia_geometrica(a, b):
    """Chamfer simétrica, mínima sobre rotaciones globales."""
    mejor = np.inf
    for giro in ANGULOS:
        d = np.abs(a[:, None] * giro - b[None, :])
        mejor = min(mejor, 0.5 * (d.min(axis=0).mean() + d.min(axis=1).mean()))
    return float(mejor)


_RESERVADAS = None


def _reservadas():
    global _RESERVADAS
    if _RESERVADAS is None:
        todas = {**desconocidas_validacion(), **desconocidas_prueba()}
        _RESERVADAS = [(p, distancia_minima(p)) for p in todas.values()]
    return _RESERVADAS


def es_reservada(puntos):
    for ref, dmin in _reservadas():
        if abs(len(puntos) - len(ref)) <= 0.25 * len(ref):
            if distancia_geometrica(puntos, ref) < 0.35 * dmin:
                return True
    return False


def _puntos_disco(rng, m):
    separacion = 1.2 / np.sqrt(m)
    puntos = []
    intentos = 0
    while len(puntos) < m:
        z = np.sqrt(rng.uniform()) * np.exp(2j * np.pi * rng.uniform())
        intentos += 1
        if all(abs(z - p) >= separacion for p in puntos):
            puntos.append(z)
        elif intentos > 200 * m:
            separacion *= 0.9
            intentos = 0
    return normalizar_puntos(puntos)


def _familia(rng):
    tipo = rng.choice(["psk", "rect", "cruz", "pam", "apsk", "hex", "disco"],
                      p=[0.15, 0.2, 0.1, 0.05, 0.2, 0.05, 0.25])
    if tipo == "psk":
        return psk(int(rng.integers(2, 33)), rng.uniform(0, np.pi))
    if tipo == "rect":
        return normalizar_puntos(rejilla(int(rng.integers(2, 12)), int(rng.integers(2, 12))))
    if tipo == "cruz":
        lado = int(rng.integers(4, 13))
        return qam_cruz(lado, int(rng.integers(1, max(2, lado // 3 + 1))))
    if tipo == "pam":
        return pam(int(rng.integers(2, 17)))
    if tipo == "apsk":
        n = int(rng.integers(2, 5))
        cuentas = [int(rng.integers(1 + 3 * i, 6 + 6 * i)) for i in range(n)]
        radios = np.cumsum(rng.uniform(0.5, 1.6, n))
        return apsk(cuentas, radios, rng.uniform(0, np.pi, n))
    if tipo == "hex":
        return hexagonal(int(rng.integers(1, 5)))
    return _puntos_disco(rng, int(rng.integers(3, 65)))


def geometria_aleatoria(rng):
    """Geometría sin etiqueta; nunca se parece a una desconocida reservada."""
    while True:
        z = _familia(rng)
        if len(z) > 128 or len(z) < 2:
            continue
        if rng.uniform() < 0.3:
            escala = rng.uniform(0.02, 0.12) * distancia_minima(z)
            z = z + escala * (rng.normal(size=len(z)) + 1j * rng.normal(size=len(z)))
        z = normalizar_puntos(z)
        if len(z) >= 2 and not es_reservada(z):
            return z


def contar_anillos(puntos, tolerancia=0.04):
    r = np.sort(np.abs(puntos - puntos.mean()))
    return int(1 + np.sum(np.diff(r) > tolerancia * max(r[-1], 1e-9)))
