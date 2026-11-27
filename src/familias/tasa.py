"""Tasa de símbolo ciega que también sirve para envolvente constante (FSK, MSK, GMSK).

CONTRATO (unidad U7, revisado; segundo intento implementado por Claude):

Dependencias permitidas (lista cerrada): numpy, src.familias.vistas.estimar_banda.

- ``estimar_tasa(iq, sps_min=2.2, sps_max=40.0) -> dict``
  - ``iq``: array complejo 1-D (complex128 tras ``nan_to_num``). Si ``len(iq) < 256`` o la potencia media es 0 o no
    finita, devolver ``{"sps": None, "puntaje": 0.0, "metodo": None}``.
  - Filtrado a la banda ocupada: ``centro, ancho = estimar_banda(x)``; bajar a cero el centro y anular por FFT las
    frecuencias con ``|f| > 0.75 * ancho``: ``y``.
  - Tres detectores, cada uno menos su media:
    ``"envolvente" = |y|^2``, ``"frecuencia" = |diff(f)|`` y ``"frecuencia2" = f^2``, con
    ``f[n] = angle(y[n+1] * conj(y[n]))``.
  - Banda de búsqueda: ``1/sps_max <= fr <= min(1/sps_min, 1.5 * ancho)`` (la tasa no puede superar el ancho ocupado).
  - Por detector: ``S = |rfft(s * hanning(L), 8L)|``; pico ``k`` = máximo de S en la banda; prominencia
    ``S[k] / mediana(S en la banda)``.
  - Detectores fuertes: prominencia >= 0.5 * la mayor. Entre ellos gana la frecuencia más baja (un armónico 2x o 3x
    suele aparecer en un solo detector). ``sps = 1 / fr[k]``, ``puntaje`` = la mayor prominencia, ``metodo`` = el
    detector elegido.

Criterio de aceptación medido (versión original fallaba en 13 de 45 casos por armónicos y en GMSK):
1. MSK, 2FSK, 4FSK y QPSK a sps 4, 8 y 16, 3 semillas, SNR 20: ``|sps_estimado / sps - 1| < 0.02`` en todos.
2. GMSK (BT sorteado en 0.25, 0.3 y 0.5): al menos 2 de las 3 semillas por tasa. Con BT 0.25 la señal casi no tiene
   rasgo cíclico y puede fallar.
3. El puntaje del ruido puro (mediana 4.1) queda por debajo del de todas las señales del punto 1, pero se solapa con
   el GMSK más débil (5.6): el puntaje no sirve como detector de presencia de señal.
4. Entradas vacías, cortas, todo ceros o con NaN no lanzan excepciones.
"""
import numpy as np

from src.familias.vistas import estimar_banda

FRACCION_FUERTE = 0.5
MARGEN_BANDA = 0.75
TOPE_TASA = 1.5


def _vacio():
    return {"sps": None, "puntaje": 0.0, "metodo": None}


def _filtrar(x):
    centro, ancho = estimar_banda(x)
    fr = np.fft.fftfreq(len(x))
    espectro = np.fft.fft(x * np.exp(-2j * np.pi * centro * np.arange(len(x))))
    espectro[np.abs(fr) > MARGEN_BANDA * ancho] = 0
    return np.fft.ifft(espectro), ancho


def _detectores(y):
    f = np.angle(y[1:] * np.conj(y[:-1]))
    return {"envolvente": np.abs(y) ** 2, "frecuencia": np.abs(np.diff(f)), "frecuencia2": f ** 2}


def _pico(s, bajo, alto):
    s = s - s.mean()
    n = 8 * len(s)
    espectro = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
    fr = np.fft.rfftfreq(n)
    banda = np.flatnonzero((fr >= bajo) & (fr <= alto))
    if not len(banda):
        return None
    mediana = float(np.median(espectro[banda]))
    if not np.isfinite(mediana) or mediana <= 0:
        return None
    k = banda[int(np.argmax(espectro[banda]))]
    return float(espectro[k] / mediana), float(fr[k])


def estimar_tasa(iq, sps_min=2.2, sps_max=40.0):
    if iq is None or not (0 < sps_min < sps_max):
        return _vacio()
    x = np.nan_to_num(np.asarray(iq, dtype=np.complex128).ravel())
    if len(x) < 256:
        return _vacio()
    potencia = float(np.mean(np.abs(x) ** 2))
    if not np.isfinite(potencia) or potencia <= 0:
        return _vacio()
    y, ancho = _filtrar(x)
    bajo, alto = 1.0 / sps_max, min(1.0 / sps_min, TOPE_TASA * ancho)
    candidatos = []
    for nombre, s in _detectores(y).items():
        r = _pico(s, bajo, alto)
        if r is not None:
            candidatos.append((*r, nombre))
    if not candidatos:
        return _vacio()
    tope = max(c[0] for c in candidatos)
    _, f0, metodo = min((c for c in candidatos if c[0] >= FRACCION_FUERTE * tope), key=lambda c: c[1])
    return {"sps": 1.0 / f0, "puntaje": tope, "metodo": metodo}
