"""Especialista dentro de la familia de frecuencia: cuenta tonos y estima el índice de modulación.

CONTRATO (unidad U8, cerrado):

Dependencias permitidas (lista cerrada): numpy, scipy.signal.find_peaks.

- ``analizar_fsk(iq, sps) -> dict`` con claves ``tonos`` (int), ``h`` (float), ``niveles`` (list[float]) y
  ``puntaje`` (float).
  - ``iq``: array complejo 1-D (complex128 tras convertir y ``nan_to_num``). ``sps``: muestras por símbolo (> 0).
  - Si ``sps`` no es finito o es <= 0, o ``len(iq) < 8 * sps + 2``, devolver
    ``{"tonos": 0, "h": 0.0, "niveles": [], "puntaje": 0.0}``.
  - Frecuencia instantánea en ciclos por símbolo: ``g[n] = angle(x[n+1] * conj(x[n])) / (2*pi) * sps``;
    peso ``w[n] = |x[n+1] * conj(x[n])|``.
  - Suavizar ``g`` con media móvil de largo ``m = max(1, round(sps / 2))`` (``np.convolve(g, ones(m)/m, "same")``):
    ``gs``.
  - Tramos estables: varianza local ``v = np.convolve((g - gs)**2, ones(m)/m, "same")``; conservar solo los índices
    con ``v <= np.percentile(v, 50)`` (las transiciones entre tonos tienen varianza alta y crean picos falsos).
  - Histograma de ``gs`` en los índices conservados, con sus pesos ``w``: 240 bins en ``[-3, 3]``. Suavizado gaussiano del histograma con
    sigma = 2 bins (núcleo truncado en ±6 bins, normalizado a suma 1, ``np.convolve(..., "same")``).
  - Picos: ``find_peaks(h, height=0.15 * max(h), distance=4)``. Si el histograma es todo cero, resultado vacío.
  - ``niveles``: centros de bin de los picos, ordenados ascendente, como floats. ``tonos = len(niveles)``
    (como máximo 16: quedarse con los 16 más altos y reordenar).
  - ``h``: mediana de ``diff(niveles)`` si hay 2 tonos o más; si no, 0.0.
  - ``puntaje``: fracción de la masa del histograma (sin suavizar) a menos de ``h / 4`` de algún nivel (0.0 si h = 0).

Criterio de aceptación (``sintetizar(Modulacion(...), "procedural", "ideal", Config(n_simbolos=2048, sps=16.0,
pulso="rrc", beta=0.35, span=10, snr_db=30.0, cfo=0.0), rng)``; las de h fijo se crean con
``Modulacion(nombre, "frecuencia", "cpm", orden=M, h=(h, h))``):
1. 2FSK con h = 1.0: tonos 2 y ``|h - 1.0| < 0.1``.
2. 4FSK con h = 0.5: tonos 4 y ``|h - 0.5| < 0.075``.
3. 3FSK con h = 0.8: tonos 3.
4. MSK del catálogo: tonos 2 y ``|h - 0.5| < 0.05``.
5. Entradas vacías, cortas, con NaN o sps inválido no lanzan excepciones.
"""
import numpy as np
from scipy.signal import find_peaks


def _vacio():
    return {"tonos": 0, "h": 0.0, "niveles": [], "puntaje": 0.0}


def _nucleo_gaussiano(sigma=2.0, radio=6):
    k = np.arange(-radio, radio + 1, dtype=np.float64)
    nucleo = np.exp(-0.5 * (k / sigma) ** 2)
    return nucleo / nucleo.sum()


def analizar_fsk(iq, sps):
    try:
        sps_f = float(sps)
    except (TypeError, ValueError):
        return _vacio()
    if not np.isfinite(sps_f) or sps_f <= 0.0 or iq is None:
        return _vacio()

    try:
        x = np.asarray(iq, dtype=np.complex128)
    except (TypeError, ValueError):
        return _vacio()
    if x.ndim != 1 or len(x) < 8 * sps_f + 2:
        return _vacio()

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    prod = x[1:] * np.conj(x[:-1])
    g = np.nan_to_num(np.angle(prod) / (2.0 * np.pi) * sps_f, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(np.abs(prod), nan=0.0, posinf=0.0, neginf=0.0)

    m = max(1, round(sps_f / 2.0))
    kernel = np.ones(m, dtype=np.float64) / m
    g_suave = np.convolve(g, kernel, "same")

    v = np.convolve((g - g_suave) ** 2, kernel, "same")
    estables = v <= np.percentile(v, 50)

    hist, bordes = np.histogram(g_suave[estables], bins=240, range=(-3.0, 3.0), weights=w[estables])
    masa_total = float(hist.sum())
    if not np.isfinite(masa_total) or masa_total <= 0.0:
        return _vacio()

    h_hist = np.convolve(hist, _nucleo_gaussiano(), "same")
    max_h = float(np.max(h_hist))
    if not np.isfinite(max_h) or max_h <= 0.0:
        return _vacio()

    picos, _ = find_peaks(h_hist, height=0.15 * max_h, distance=4)
    if len(picos) == 0:
        return _vacio()

    if len(picos) > 16:
        top = np.argsort(h_hist[picos])[-16:]
        picos = np.sort(picos[top])

    centros = 0.5 * (bordes[:-1] + bordes[1:])
    niveles = [float(centros[i]) for i in picos]
    tonos = len(niveles)
    h_est = float(np.median(np.diff(niveles))) if tonos >= 2 else 0.0

    if h_est > 0.0:
        niveles_arr = np.asarray(niveles, dtype=np.float64)
        dist_min = np.min(np.abs(centros[:, None] - niveles_arr[None, :]), axis=1)
        puntaje = float(hist[dist_min < (h_est / 4.0)].sum() / masa_total)
    else:
        puntaje = 0.0

    return {"tonos": tonos, "h": h_est, "niveles": niveles, "puntaje": puntaje}
