"""Etapa 3: lista conforme («también puede ser») y equivalencias («misma señal que»).

CONTRATO (unidad U9, cerrado):

Dependencias permitidas (lista cerrada): numpy.

- ``class ConjuntoConforme``
  - ``__init__(self, alfa=0.1)``: si ``not 0 < alfa < 1`` lanzar ``ValueError``. Atributos ``alfa`` y ``corte``
    (None hasta ajustar).
  - ``ajustar(self, p, y) -> self``: ``p`` array (N, K) de probabilidades, ``y`` array (N,) de enteros en [0, K).
    ``ValueError`` si N == 0, si ``len(y) != N`` o si algún ``y`` está fuera de rango.
    Puntajes ``s_i = 1 - p[i, y_i]``. ``nivel = min(1, ceil((N + 1) * (1 - alfa)) / N)``.
    ``corte = np.quantile(s, nivel, method="higher")`` (float).
  - ``conjunto(self, p) -> list[list[int]]``: ``RuntimeError`` si no se ajustó. ``p`` (K,) o (N, K); se trata como
    (N, K). Por fila: índices k con ``p[k] >= 1 - corte``; si queda vacío, ``[argmax]``. Ordenados por probabilidad
    descendente (empates por índice ascendente). Devuelve siempre una lista de listas de int nativos.

- ``salida_final(p, nombres, conjunto, equivalentes, umbral, indeterminada="INDETERMINADA") -> dict``
  - ``p``: array (K,) de probabilidades; ``nombres``: lista de K nombres; ``conjunto``: lista de índices (la de
    ``ConjuntoConforme.conjunto`` para esa fila); ``equivalentes``: dict nombre -> tupla de nombres; ``umbral``: float.
  - ``k = argmax(p)``. ``modulacion = nombres[k]`` si ``p[k] >= umbral``, si no ``indeterminada``.
  - Devuelve ``{"modulacion": ..., "confianza": float(p[k]), "mas_probable": nombres[k],
    "misma_senal_que": list(equivalentes.get(nombres[k], ())),
    "tambien_puede_ser": [{"nombre": nombres[j], "probabilidad": float(p[j])} for j in conjunto if j != k]}``.

Criterio de aceptación:
1. Datos sintéticos: K = 6; ``y`` uniforme; ``p = Dirichlet(alpha=1 + 6*onehot(y))`` con ``rng = default_rng(0)``.
   Ajustar con 5000 filas; en otras 20000 filas, la cobertura (``y_i`` dentro de su conjunto) está a ±0.02 de
   ``max(1 - alfa, exactitud del argmax)`` para ``alfa`` en (0.05, 0.1, 0.2). (Revisión: el conjunto nunca queda vacío
   porque incluye el argmax, así que la cobertura no puede bajar de la exactitud del argmax, 0,962 con estos datos;
   el criterio original pedía 0,80 y 0,90, imposible con una implementación fiel al contrato.)
2. Una fila con ``p = [0.97, 0.01, 0.01, 0.01, 0, 0]`` da conjunto ``[0]`` con ``alfa = 0.1`` en los datos del punto 1.
3. ``salida_final`` con nombres ``["BPSK", "QPSK"]``, ``p = [0.3, 0.7]``, conjunto ``[1, 0]``,
   ``equivalentes = {"QPSK": ("DQPSK",)}`` y umbral 0.6 da modulación "QPSK", misma_senal_que ["DQPSK"] y
   tambien_puede_ser ``[{"nombre": "BPSK", "probabilidad": 0.3}]``; con umbral 0.8, modulación "INDETERMINADA".
4. Los errores descritos lanzan las excepciones indicadas.
"""
import numpy as np


class ConjuntoConforme:
    def __init__(self, alfa=0.1):
        if isinstance(alfa, bool) or not 0 < alfa < 1:
            raise ValueError(f"alfa debe estar en (0, 1), recibido: {alfa!r}")
        self.alfa = float(alfa)
        self.corte = None

    def ajustar(self, p, y):
        p_arr = np.asarray(p, dtype=float)
        if p_arr.ndim != 2:
            raise ValueError(f"p debe tener forma (N, K), recibido ndim={p_arr.ndim}")
        n, k = p_arr.shape
        if n == 0:
            raise ValueError("p no puede estar vacío (N == 0)")
        if k == 0:
            raise ValueError("p debe tener al menos una clase (K > 0)")

        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or len(y_arr) != n:
            raise ValueError(f"y debe tener forma ({n},), recibido {y_arr.shape}")
        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError(f"y debe contener enteros, recibido dtype={y_arr.dtype}")
        if np.any((y_arr < 0) | (y_arr >= k)):
            raise ValueError(f"y contiene valores fuera de [0, {k})")

        s = 1.0 - p_arr[np.arange(n), y_arr]
        nivel = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alfa))) / n)
        self.corte = float(np.quantile(s, nivel, method="higher"))
        return self

    def conjunto(self, p):
        if self.corte is None:
            raise RuntimeError("ConjuntoConforme debe ajustarse antes de llamar a conjunto()")
        p_arr = np.asarray(p, dtype=float)
        if p_arr.ndim == 1:
            p_arr = p_arr[np.newaxis, :]
        elif p_arr.ndim != 2:
            raise ValueError(f"p debe tener forma (K,) o (N, K), recibido ndim={p_arr.ndim}")
        if p_arr.shape[1] == 0:
            raise ValueError("p debe tener al menos una clase (K > 0)")

        umbral_p = 1.0 - self.corte
        mascara = p_arr >= umbral_p
        resultado = []
        for fila, m in zip(p_arr, mascara):
            idx = np.flatnonzero(m)
            if idx.size == 0:
                resultado.append([int(np.argmax(fila))])
            elif idx.size == 1:
                resultado.append([int(idx[0])])
            else:
                orden = np.argsort(-fila[idx], kind="stable")
                resultado.append([int(i) for i in idx[orden]])
        return resultado


def salida_final(p, nombres, conjunto, equivalentes, umbral, indeterminada="INDETERMINADA"):
    p_arr = np.asarray(p, dtype=float)
    if p_arr.ndim != 1 or p_arr.size == 0:
        raise ValueError(f"p debe tener forma (K,) con K > 0, recibido {p_arr.shape}")
    if len(nombres) != p_arr.size:
        raise ValueError(f"nombres debe tener longitud {p_arr.size}, recibido {len(nombres)}")

    k = int(np.argmax(p_arr))
    confianza = float(p_arr[k])
    mas_probable = nombres[k]
    modulacion = mas_probable if confianza >= umbral else indeterminada
    return {
        "modulacion": modulacion,
        "confianza": confianza,
        "mas_probable": mas_probable,
        "misma_senal_que": list(equivalentes.get(mas_probable, ())),
        "tambien_puede_ser": [
            {"nombre": nombres[j], "probabilidad": float(p_arr[j])}
            for j in conjunto
            if j != k
        ],
    }
