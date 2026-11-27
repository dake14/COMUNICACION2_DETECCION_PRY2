"""Seguimiento de canales que cambian: CMA símbolo a símbolo y corrección de fase por bloques.

CONTRATO (unidad U3, cerrado):

Dependencias permitidas (lista cerrada): numpy, src.cascada.clasica.ventanas_t2.

- ``cma_seguimiento(x2, pesos, mu=1e-3) -> tuple[np.ndarray, np.ndarray]``
  - ``x2``: array complejo 1-D de muestras a dos por símbolo. ``pesos``: array complejo de largo L (17 en el proyecto).
  - Normalizar ``x2`` a potencia media 1. ``v = ventanas_t2(x2, len(pesos))`` (matriz K x L).
  - Si K == 0 devolver ``(np.zeros(0, complex128), pesos.astype(complex128))``.
  - ``y0 = v @ pesos``; módulo de referencia ``R = mean(|y0|^4) / mean(|y0|^2)`` (si el denominador es 0, R = 1).
  - Recorrer k = 0..K-1 en orden: ``y = v[k] @ w``; ``e = y * (|y|^2 - R)``; ``w = w - mu * e * conj(v[k])``;
    guardar y. ``w`` arranca como copia complex128 de ``pesos``.
  - Si en algún momento ``w`` o ``y`` dejan de ser finitos, devolver ``(y0, pesos.astype(complex128))``.
  - Devolver (salidas complex128 de largo K, w final). No modifica ``pesos`` ni ``x2`` del llamador.

- ``fase_por_bloques(z, bloque=64, bins=90) -> np.ndarray``
  - Bloques consecutivos sin solape de ``bloque`` símbolos; el último incompleto se une al anterior.
    Si hay menos de 2 bloques, devolver ``z`` como complex128 sin cambios.
  - Histograma de ángulos de cada bloque: ``np.histogram(np.angle(zb), bins=bins, range=(-pi, pi), weights=|zb|^2)``,
    suavizado circular con el núcleo [0.25, 0.5, 0.25] (``np.roll``), normalizado a suma 1.
  - Primera pasada: correlación ``c_b(s) = sum(h_{b-1} * np.roll(h_b, -s))`` entre bloques vecinos para s entero en
    [-bins//8, bins//8]. Los saltos s se eligen con programación dinámica sobre todos los bloques (log c menos una
    penalización ``1e-3 s^2`` y ``0.05 max(0, |s - s'| - 1)^2`` entre saltos consecutivos), refinados con
    interpolación parabólica; ``theta`` es la suma acumulada.
  - Segunda pasada (con 3 bloques o más): referencia global = histograma de toda la señal corregida con la primera
    pasada; cada bloque, prerrotado con la predicción ``theta_{b-1} + salto_b``, se alinea a esa referencia en
    ±min(bins//8, 3) bins con refinamiento parabólico. La fase se fija con ``theta_0 = 0``.
  - (Revisión del contrato: la referencia "solo bloque anterior" de la versión original resbalaba entre lóbulos en
    16QAM; la implementación de agy con DP y segunda pasada pasa el criterio en 50 semillas de 16QAM, QPSK, 8PSK y
    BPSK con deriva lineal y senoidal.)
  - Fase por símbolo: ``np.interp`` lineal de theta en los centros de bloque (índice medio de cada bloque) sobre
    todos los índices de símbolo (constante fuera de los extremos). Devolver ``z * exp(-j fase)`` complex128.

Criterio de aceptación:
1. 16QAM de 4000 símbolos con fase que crece 0.004 rad por símbolo (y SNR 25 dB): tras ``fase_por_bloques`` la
   fase residual (respecto al ángulo correcto, quitando una constante) tiene desvío < 0.1 rad.
2. QPSK por un canal de 2 coeficientes que cambia lentamente: el error |(|y|^2 - 1)| medio de la última mitad de
   ``cma_seguimiento`` es menor que el de la salida con pesos fijos.
3. Entradas vacías o cortas no lanzan excepciones.
"""
import numpy as np

from src.cascada.clasica import ventanas_t2


def _hist_angulos(zb, bins):
    pesos = np.abs(zb) ** 2
    h, _ = np.histogram(np.angle(zb), bins=bins, range=(-np.pi, np.pi), weights=pesos)
    h = 0.25 * np.roll(h, 1) + 0.5 * h + 0.25 * np.roll(h, -1)
    total = float(h.sum())
    if total > 0.0:
        h = h / total
    return h


def _refinar_pico(c, idx, shifts):
    s_star = shifts[idx]
    if 0 < idx < len(shifts) - 1:
        c_menos, c_cero, c_mas = c[idx - 1], c[idx], c[idx + 1]
        denom = c_menos - 2.0 * c_cero + c_mas
        d = 0.5 * (c_menos - c_mas) / denom if denom != 0.0 else 0.0
    else:
        d = 0.0
    return float(s_star + d)


def cma_seguimiento(x2, pesos, mu=1e-3):
    x2_arr = np.asarray(x2, dtype=np.complex128)
    w0 = np.array(pesos, dtype=np.complex128, copy=True)
    largo = len(w0)
    if largo == 0 or len(x2_arr) < largo:
        return np.zeros(0, dtype=np.complex128), w0

    pot = float(np.mean(np.abs(x2_arr) ** 2))
    if pot > 0.0 and np.isfinite(pot):
        x2_norm = x2_arr / np.sqrt(pot)
    else:
        x2_norm = x2_arr

    v = ventanas_t2(x2_norm, largo)
    k_total = len(v)
    if k_total == 0:
        return np.zeros(0, dtype=np.complex128), w0

    y0 = np.asarray(v @ w0, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):
        abs2 = np.abs(y0) ** 2
        denom = float(np.mean(abs2))
        r_ref = float(np.mean(abs2 ** 2) / denom) if denom != 0.0 else 1.0
    if not np.isfinite(r_ref):
        return y0, w0

    w = w0.copy()
    salidas = np.empty(k_total, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(k_total):
            vk = v[k]
            y = vk @ w
            if not np.isfinite(y):
                return y0, w0
            e = y * (y.real * y.real + y.imag * y.imag - r_ref)
            w = w - mu * e * np.conj(vk)
            if not np.all(np.isfinite(w)):
                return y0, w0
            salidas[k] = y

    return salidas, w


def fase_por_bloques(z, bloque=64, bins=90):
    z_arr = np.asarray(z, dtype=np.complex128)
    n = len(z_arr)
    if bloque <= 0 or bins <= 0 or n // bloque < 2:
        return np.array(z_arr, dtype=np.complex128, copy=True)

    nb = n // bloque
    slices = [
        (b * bloque, (b + 1) * bloque if b < nb - 1 else n)
        for b in range(nb)
    ]
    r = bins // 8
    shifts = np.arange(-r, r + 1)
    m = len(shifts)
    paso_rad = 2.0 * np.pi / bins

    hists = [_hist_angulos(z_arr[a:b], bins) for a, b in slices]
    corrs = np.empty((nb - 1, m), dtype=np.float64)
    for b in range(1, nb):
        h_prev = hists[b - 1]
        h_b = hists[b]
        for i, s in enumerate(shifts):
            corrs[b - 1, i] = np.sum(h_prev * np.roll(h_b, -s))

    log_c = np.log(np.maximum(corrs, 1e-12)) - 1e-3 * (shifts ** 2)[None, :]
    dp = np.empty_like(log_c)
    back = np.zeros((nb - 1, m), dtype=np.int64)
    dp[0] = log_c[0]
    pen = 0.05 * (np.maximum(0, np.abs(shifts[:, None] - shifts[None, :]) - 1) ** 2)
    for t in range(1, nb - 1):
        scores = dp[t - 1][:, None] - pen
        best_prev = np.argmax(scores, axis=0)
        back[t] = best_prev
        dp[t] = log_c[t] + scores[best_prev, np.arange(m)]

    path = np.empty(nb - 1, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(nb - 3, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    deltas = np.array(
        [_refinar_pico(corrs[b - 1], int(path[b - 1]), shifts) * paso_rad for b in range(1, nb)],
        dtype=np.float64,
    )
    thetas1 = np.zeros(nb, dtype=np.float64)
    thetas1[1:] = np.cumsum(deltas)

    centros = np.array([0.5 * (a + b - 1) for a, b in slices], dtype=np.float64)
    if nb >= 3:
        fase1 = np.interp(np.arange(n, dtype=np.float64), centros, thetas1)
        h_ref = _hist_angulos(z_arr * np.exp(-1j * fase1), bins)
        r2 = min(r, 3)
        shifts2 = np.arange(-r2, r2 + 1)
        thetas = np.zeros(nb, dtype=np.float64)
        a0, b0 = slices[0]
        h0 = _hist_angulos(z_arr[a0:b0], bins)
        c0 = np.array([np.sum(h_ref * np.roll(h0, -s)) for s in shifts2], dtype=np.float64)
        thetas[0] = _refinar_pico(c0, int(np.argmax(c0)), shifts2) * paso_rad
        for b in range(1, nb):
            a1, b1 = slices[b]
            pred = thetas[b - 1] + deltas[b - 1]
            h_b = _hist_angulos(z_arr[a1:b1] * np.exp(-1j * pred), bins)
            c2 = np.array([np.sum(h_ref * np.roll(h_b, -s)) for s in shifts2], dtype=np.float64)
            thetas[b] = pred + _refinar_pico(c2, int(np.argmax(c2)), shifts2) * paso_rad
        thetas = thetas - thetas[0]
    else:
        thetas = thetas1

    fase = np.interp(np.arange(n, dtype=np.float64), centros, thetas)
    return (z_arr * np.exp(-1j * fase)).astype(np.complex128)
