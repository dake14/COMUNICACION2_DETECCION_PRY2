"""Vistas finas de una constelación recuperada, para los especialistas dentro de cada familia.

CONTRATO (unidad U13, cerrado):

Dependencias permitidas (lista cerrada): numpy.

- ``BINS = 256``, ``FORMA = (4, BINS)``.
- ``vistas_constelacion(simbolos) -> np.ndarray`` de forma ``FORMA`` y dtype ``float32``.
  - ``simbolos``: array complejo 1-D (convertir a complex128 y ``nan_to_num``; si ``None`` o no es convertible, tratarlo
    como vacío).
  - Si quedan menos de 32 símbolos, o la potencia media ``mean(|s|^2)`` no es finita o es <= 0: devolver
    ``np.zeros(FORMA, np.float32)``.
  - Normalizar: ``s = s / sqrt(mean(|s|^2))``.
  - Alinear con los ejes: ``m4 = mean(s**4)``; si ``|m4| > 1e-3``, ``theta = (angle(m4) - pi) / 4`` y
    ``s = s * exp(-1j * theta)`` (una QAM cuadrada queda con la rejilla sobre los ejes); si no, no rotar.
  - Canales, cada uno con ``np.histogram(..., bins=BINS, range=...)``:
    0. ``real(s)`` en ``[-1.8, 1.8]``.
    1. ``imag(s)`` en ``[-1.8, 1.8]``.
    2. ``abs(s)`` en ``[0, 2]``.
    3. ``angle(s)`` en ``[-pi, pi]``.
  - Cada canal se divide por su máximo si el máximo es > 0 (queda en [0, 1]).

Criterio de aceptación (símbolos sintéticos con ``rng = np.random.default_rng(0)``, 4000 símbolos, rotados 0.3 rad y
con ruido gaussiano complejo de SNR 30 dB; picos contados con ``scipy.signal.find_peaks(canal, height=0.2,
distance=3)``):
1. 16QAM (niveles ±1, ±3 en I y Q): 4 picos en los canales 0 y 1.
2. 64QAM (niveles ±1, ±3, ±5, ±7): 8 picos en los canales 0 y 1.
3. 8PSK sobre el círculo unidad: el máximo del canal 2 está a ±3 bins del índice 128 (radio 1) y el canal 3 tiene 8
   picos.
4. 16APSK (4 puntos en radio 1 y 12 en radio 2.7, en ese orden de fases equiespaciadas): 2 picos en el canal 2.
5. La salida es ``float32`` de forma ``(4, 256)``, sin NaN, con valores en [0, 1] y máximo 1 en cada canal.
6. Entradas vacías, de 10 símbolos, todo NaN, todo ceros o ``None`` devuelven ceros sin lanzar excepciones.
"""
import numpy as np

BINS = 256
FORMA = (4, BINS)


def vistas_constelacion(simbolos):
    if simbolos is None:
        return np.zeros(FORMA, np.float32)
    try:
        s = np.asarray(simbolos, dtype=np.complex128).ravel()
    except (TypeError, ValueError):
        return np.zeros(FORMA, np.float32)

    s = np.nan_to_num(s)
    if s.size < 32:
        return np.zeros(FORMA, np.float32)

    with np.errstate(over="ignore", invalid="ignore"):
        potencia = float(np.mean(np.abs(s) ** 2))
    if not np.isfinite(potencia) or potencia <= 0.0:
        return np.zeros(FORMA, np.float32)

    s = s / np.sqrt(potencia)
    m4 = np.mean(s ** 4)
    if np.abs(m4) > 1e-3:
        theta = (np.angle(m4) - np.pi) / 4.0
        s = s * np.exp(-1j * theta)

    canales = (
        (np.real(s), (-1.8, 1.8)),
        (np.imag(s), (-1.8, 1.8)),
        (np.abs(s), (0.0, 2.0)),
        (np.angle(s), (-np.pi, np.pi)),
    )
    salida = np.zeros(FORMA, np.float32)
    for idx, (valores, rango) in enumerate(canales):
        hist, _ = np.histogram(valores, bins=BINS, range=rango)
        maximo = hist.max()
        if maximo > 0:
            salida[idx] = (hist / maximo).astype(np.float32)
    return salida
