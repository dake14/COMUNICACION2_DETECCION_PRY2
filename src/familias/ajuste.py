"""¿Los símbolos recuperados caben en la constelación de una clase conocida? Prueba física, sin entrenar.

CONTRATO (unidad U14, cerrado):

Dependencias permitidas (lista cerrada): numpy, scipy.spatial.cKDTree, scipy.cluster.vq.kmeans2, warnings.

Constantes: ``N_MAX = 2048``, ``MIN_SIMBOLOS = 256``, ``K_LIBRE = 64``, ``GIROS = 48``.

- ``preparar(simbolos) -> np.ndarray | None``
  - ``None`` o algo no convertible a array complejo 1-D (``TypeError``/``ValueError``) -> ``None``.
  - Convertir a complex128, ``ravel``, ``nan_to_num``; quitar los ceros exactos (``s != 0``); quedarse con los primeros
    ``N_MAX``.
  - Si quedan menos de ``MIN_SIMBOLOS`` o la potencia media ``mean(|s|^2)`` no es finita o es <= 0 -> ``None``.
  - Devolver ``s / sqrt(mean(|s|^2))``.
- ``error_libre(s) -> float``: ajuste libre de referencia.
  - ``z = np.c_[s.real, s.imag]``; ``centros, etiquetas = kmeans2(z, K_LIBRE, iter=30, minit="++",
    rng=np.random.default_rng(0))`` con las advertencias silenciadas (``warnings.catch_warnings()`` +
    ``simplefilter("ignore")``).
  - Devolver ``mean(sum((z - centros[etiquetas])**2, axis=1)) / mean(|s|^2)`` como ``float``.
- ``ajuste_plantilla(s, puntos) -> tuple[float, float]``: ``(error, ocupacion)``.
  - ``p = asarray(puntos, complex128)``; ``p = p / sqrt(mean(|p|^2))``. Árbol: ``cKDTree(np.c_[p.real, p.imag])``.
  - Para cada ángulo ``theta`` de ``np.linspace(0, 2*pi, GIROS, endpoint=False)`` (en ese orden):
    - ``z = s * exp(1j*theta)``.
    - Tres veces: ``i`` = índice del punto más cercano a cada ``z``; ``d = p[i]``; ``energia = vdot(d, d).real``; si
      ``energia <= 1e-9`` salir del bucle; ``a = vdot(d, z) / energia``; si ``|a| < 1e-3`` salir del bucle;
      ``z = z / a``.
    - ``i`` = más cercano otra vez; ``error = mean(|z - p[i]|^2) / mean(|z|^2)``.
    - Guardar el ángulo de menor ``error`` (en empate, el primero) y ``cuentas = bincount(i, minlength=len(p))`` de ese
      ángulo.
  - ``c = cuentas / len(s)``, solo los ``c > 0``; ``ocupacion = exp(-sum(c * log(c))) / len(p)`` (número efectivo de
    puntos usados, en fracción de la plantilla).
  - Devolver ``(float(error), float(ocupacion))``.
- ``medir(simbolos, plantillas) -> dict | None``
  - ``plantillas``: dict clase -> puntos complejos. ``s = preparar(simbolos)``; si es ``None`` devolver ``None``.
  - Devolver ``{"libre": error_libre(s), "plantillas": {clase: {"error": e, "ocupacion": o}}}`` con ``(e, o) =
    ajuste_plantilla(s, puntos)`` para cada clase, en el orden de ``plantillas``.
- ``class Consistencia``: umbrales por clase, calibrados solo con archivos conocidos de esa misma clase.
  - ``__init__(self, beta=0.05)``: guarda ``self.beta = float(beta)`` y ``self.umbrales = {}``. ``ValueError`` si
    ``beta`` no está en ``(0, 0.5)``.
  - ``razon(medicion, clase) -> float``: ``log(error / libre)`` de esa clase; ``nan`` si ``libre <= 0`` o
    ``error <= 0``.
  - ``ajustar(self, mediciones, clases) -> Consistencia``
    - ``mediciones``: lista de dicts de ``medir`` o ``None``; ``clases``: lista de str del mismo largo
      (``ValueError`` si los largos difieren).
    - Para cada clase ``k`` que aparece en las plantillas de alguna medición no ``None``: tomar las mediciones no
      ``None`` cuya clase verdadera es ``k``. Si son menos de 10, no hay umbral para ``k``. Si no:
      ``self.umbrales[k] = {"razon": float(nanquantile(razones, 1 - beta)), "ocupacion": float(quantile(ocupaciones,
      beta))}``. Si todas las razones son ``nan``, ``"razon"`` es ``inf``.
    - Devuelve ``self``.
  - ``consistentes(self, medicion) -> dict``: clase -> bool para cada clase de ``self.umbrales``.
    - ``medicion is None``: todas ``True``.
    - Si no: ``True`` si la clase no está en ``medicion["plantillas"]``; si está, ``True`` si (``razon`` es ``nan`` o
      ``razon <= umbral["razon"]``) y ``ocupacion >= umbral["ocupacion"]``.

Criterio de aceptación (símbolos sintéticos: ``rng = np.random.default_rng(0)``; puntos normalizados a potencia 1,
2048 símbolos al azar, multiplicados por ``1.7 * exp(0.3j)`` y con ruido complejo gaussiano de SNR 25 dB salvo que se
diga otra cosa; ``g = src.cascada.geometria``; ``razon = log(error / error_libre)`` sobre ``preparar(...)``):
1. 16QAM (``g.qam(16)``): con la plantilla ``g.qam(16)`` razón < 1.5 y ocupación > 0.9; con ``g.psk(8)`` razón > 3.5.
2. 12PSK (``g.psk(12)``) con la plantilla ``g.psk(8)``: razón > 3.0. 8PSK con ``g.psk(8)``: razón < 2.2.
3. A SNR 30 dB: 256QAM (``g.qam(256)``) con ``g.qam(256)`` ocupación > 0.9; 128QAM (``g.qam_cruz(12, 2)``) con
   ``g.qam(256)`` ocupación < 0.8.
4. ``preparar`` devuelve ``None`` con ``None``, 100 símbolos, todo NaN o todo ceros; ``medir`` devuelve ``None`` en
   esos casos. Con 300 símbolos no devuelve ``None`` y la potencia media del resultado es 1 (±1e-9).
5. Dos llamadas a ``medir`` con la misma entrada dan exactamente el mismo resultado.
6. ``Consistencia(0.1).ajustar`` con 20 mediciones a mano de clase ``"A"`` (``libre = 1``, ``plantillas = {"A":
   {"error": exp(k/2), "ocupacion": 1 - k/100}}`` para k = 0..19) y 5 de clase ``"B"`` (``plantillas = {"A": ...,
   "B": ...}``): umbral de ``"A"`` con razón = nanquantile de k/2 al 0.9 y ocupación = quantile al 0.1; sin umbral
   para ``"B"``. ``consistentes(None)`` es ``{"A": True}``; una medición con razón 100 para ``"A"`` da
   ``{"A": False}``. ``Consistencia(0.6)`` y largos distintos lanzan ``ValueError``.
"""
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

N_MAX = 2048
MIN_SIMBOLOS = 256
K_LIBRE = 64
GIROS = 48


def preparar(simbolos):
    if simbolos is None:
        return None
    try:
        s = np.asarray(simbolos, dtype=np.complex128).ravel()
    except (TypeError, ValueError):
        return None
    s = np.nan_to_num(s)
    s = s[s != 0][:N_MAX]
    if len(s) < MIN_SIMBOLOS:
        return None
    potencia = float(np.mean(np.abs(s) ** 2))
    if not np.isfinite(potencia) or potencia <= 0.0:
        return None
    return s / np.sqrt(potencia)


def error_libre(s):
    z = np.c_[s.real, s.imag]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centros, etiquetas = kmeans2(
            z, K_LIBRE, iter=30, minit="++", rng=np.random.default_rng(0)
        )
    return float(np.mean(np.sum((z - centros[etiquetas]) ** 2, axis=1)) / np.mean(np.abs(s) ** 2))


def ajuste_plantilla(s, puntos):
    p = np.asarray(puntos, dtype=np.complex128)
    p = p / np.sqrt(np.mean(np.abs(p) ** 2))
    arbol = cKDTree(np.c_[p.real, p.imag])

    mejor_error = np.inf
    mejores_cuentas = None
    for theta in np.linspace(0.0, 2.0 * np.pi, GIROS, endpoint=False):
        z = s * np.exp(1j * theta)
        for _ in range(3):
            _, i = arbol.query(np.c_[z.real, z.imag])
            d = p[i]
            energia = np.vdot(d, d).real
            if energia <= 1e-9:
                break
            a = np.vdot(d, z) / energia
            if abs(a) < 1e-3:
                break
            z = z / a
        _, i = arbol.query(np.c_[z.real, z.imag])
        error = float(np.mean(np.abs(z - p[i]) ** 2) / np.mean(np.abs(z) ** 2))
        if error < mejor_error:
            mejor_error = error
            mejores_cuentas = np.bincount(i, minlength=len(p))

    if mejores_cuentas is None:
        mejores_cuentas = np.zeros(len(p), dtype=int)
    c = mejores_cuentas / len(s)
    c = c[c > 0]
    ocupacion = float(np.exp(-np.sum(c * np.log(c))) / len(p))
    return (float(mejor_error), ocupacion)


def medir(simbolos, plantillas):
    s = preparar(simbolos)
    if s is None:
        return None
    resultado_plantillas = {}
    for clase, puntos in plantillas.items():
        e, o = ajuste_plantilla(s, puntos)
        resultado_plantillas[clase] = {"error": e, "ocupacion": o}
    return {
        "libre": error_libre(s),
        "plantillas": resultado_plantillas,
    }


class Consistencia:
    def __init__(self, beta=0.05):
        beta = float(beta)
        if not (0.0 < beta < 0.5):
            raise ValueError("beta debe estar en (0, 0.5)")
        self.beta = beta
        self.umbrales = {}

    @staticmethod
    def razon(medicion, clase):
        libre = medicion["libre"]
        error = medicion["plantillas"][clase]["error"]
        if libre <= 0 or error <= 0:
            return float("nan")
        return float(np.log(error / libre))

    def ajustar(self, mediciones, clases):
        mediciones = list(mediciones)
        clases = list(clases)
        if len(mediciones) != len(clases):
            raise ValueError("mediciones y clases deben tener el mismo largo")

        self.umbrales = {}
        clases_plantilla = []
        vistas = set()
        for m in mediciones:
            if m is not None:
                for k in m["plantillas"]:
                    if k not in vistas:
                        vistas.add(k)
                        clases_plantilla.append(k)

        for k in clases_plantilla:
            muestras = [
                m for m, c in zip(mediciones, clases)
                if m is not None and c == k and k in m["plantillas"]
            ]
            if len(muestras) < 10:
                continue
            razones = np.asarray([self.razon(m, k) for m in muestras], dtype=np.float64)
            ocupaciones = np.asarray(
                [m["plantillas"][k]["ocupacion"] for m in muestras], dtype=np.float64
            )
            if np.all(np.isnan(razones)):
                umbral_razon = float("inf")
            else:
                umbral_razon = float(np.nanquantile(razones, 1.0 - self.beta))
            umbral_ocupacion = float(np.quantile(ocupaciones, self.beta))
            self.umbrales[k] = {
                "razon": umbral_razon,
                "ocupacion": umbral_ocupacion,
            }
        return self

    def consistentes(self, medicion):
        if medicion is None:
            return {clase: True for clase in self.umbrales}
        plantillas = medicion["plantillas"]
        salida = {}
        for clase, umbral in self.umbrales.items():
            if clase not in plantillas:
                salida[clase] = True
            else:
                r = self.razon(medicion, clase)
                oc = plantillas[clase]["ocupacion"]
                ok_razon = np.isnan(r) or r <= umbral["razon"]
                ok_ocupacion = oc >= umbral["ocupacion"]
                salida[clase] = bool(ok_razon and ok_ocupacion)
        return salida
