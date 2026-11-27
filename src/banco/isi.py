"""Regulador de ISI de los canales con multitrayecto.

Contrato (U17)
--------------
El nivel de ISI se expresa como la relación trayecto principal / ecos en dB: la potencia del rayo más fuerte dividida
por la suma de las potencias de todos los demás. 30 dB es casi sin ecos; 0 dB es un eco total tan fuerte como el
trayecto directo. ``None`` significa "sin regular": el canal se queda con los ecos que sorteó.

El nivel vigente vive en una ``contextvars.ContextVar`` llamada ``_NIVEL`` (por defecto ``None``) para que los
canales, que reciben solo ``(x, sps, rng)``, lo lean sin cambiar su firma. ``sintetizar`` lo fija con ``nivel_isi``.

Funciones:
- ``nivel_isi(db)``: context manager. Fija ``_NIVEL`` a ``db`` (float o None) mientras dura el ``with`` y lo
  restaura al salir, también si hay excepción (``token = _NIVEL.set(db)`` / ``_NIVEL.reset(token)``).
  ``ValueError`` si ``db`` no es None y no es finito.
- ``nivel_actual() -> float | None``: el valor de ``_NIVEL``.
- ``relacion_db(magnitudes) -> float``: con p = |magnitudes|**2 (array 1-D real o complejo, largo >= 1),
  principal = índice de argmax(p) (el primero si hay empate); devuelve 10*log10(p[principal] / suma del resto).
  ``inf`` si el resto suma 0 (incluye largo 1). ``ValueError`` si el array está vacío o p[principal] == 0.
- ``ajustar(magnitudes, db=...) -> np.ndarray``: ``db`` por defecto es el centinela ``_VIGENTE``, que significa
  "usar ``nivel_actual()``". Si el nivel resultante es None, devuelve ``np.array(magnitudes, copy=True)`` sin cambios.
  Si no: principal como en ``relacion_db``; ``resto = suma de p sin el principal``. Si ``resto == 0`` devuelve la
  copia sin cambios. Si no, multiplica todos los elementos que no son el principal por
  ``alfa = sqrt(p[principal] / (resto * 10**(db/10)))`` (el principal no cambia) y después normaliza todo el vector
  para que conserve la energía original ``sum(p)``. Conserva el dtype de entrada (real o complejo) y la fase de cada
  elemento. Resultado: ``relacion_db(salida) == db`` con tolerancia 1e-9 y ``sum(|salida|**2) == sum(p)`` con
  tolerancia relativa 1e-12. ``ValueError`` si ``db`` no es finito o el array está vacío.

Dependencias permitidas: ``contextlib``, ``contextvars``, ``numpy``. Nada más.

Criterios de aceptación:
1. ``relacion_db([1, 0.5])`` == 10*log10(4) (≈ 6.0206).
2. ``relacion_db([0.2, 1.0, 0.1])`` usa el índice 1 como principal: 10*log10(1 / 0.05).
3. ``relacion_db([1.0])`` es ``inf``; ``relacion_db([])`` y ``relacion_db([0, 0])`` levantan ``ValueError``.
4. Para m = [0.6, 0.5, 0.4, 0.3] y db en (0, 3, 10, 20, 30): ``relacion_db(ajustar(m, db))`` == db (1e-9) y la
   energía se conserva; ``ajustar(m, db)[0] / ajustar(m, db)[1]`` cambia, pero ``ajustar(m, db)[1] / ajustar(m, db)[2]``
   == 0.5/0.4 (los ecos conservan sus proporciones).
5. Con taps complejos ``t = [0.3j, 1+0j, 0.5*exp(1j)]``: la salida es compleja, ``np.angle`` de cada elemento igual al
   de la entrada, y ``relacion_db`` == db.
6. ``ajustar(m)`` fuera de un ``with`` devuelve una copia igual a m (no el mismo objeto); dentro de
   ``with nivel_isi(12.0)`` da ``relacion_db == 12``; después del ``with`` ``nivel_actual()`` vuelve a None, también
   si dentro del ``with`` se levantó una excepción.
7. ``ajustar(m, None)`` devuelve la copia sin cambios aunque haya un nivel vigente.
8. ``nivel_isi(float("nan"))`` y ``ajustar(m, float("inf"))`` levantan ``ValueError``.
"""
# ─────────────────────────────  Imports  ─────────────────────────────
import contextlib
import contextvars

import numpy as np

# ─────────────────────────────  Config  ──────────────────────────────
_NIVEL = contextvars.ContextVar("nivel_isi_db", default=None)
_VIGENTE = object()


# ─────────────────────────────  Nivel  ───────────────────────────────
@contextlib.contextmanager
def nivel_isi(db):
    if db is not None:
        db = float(db)
        if not np.isfinite(db):
            raise ValueError("db debe ser finito o None")
    token = _NIVEL.set(db)
    try:
        yield
    finally:
        _NIVEL.reset(token)


def nivel_actual():
    return _NIVEL.get()


# ─────────────────────────────  Perfil  ──────────────────────────────
def relacion_db(magnitudes):
    a = np.asarray(magnitudes)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("magnitudes debe ser un array 1-D no vacío")
    p = np.abs(a.astype(np.complex128 if np.iscomplexobj(a) else np.float64, copy=False)) ** 2
    principal = int(np.argmax(p))
    if p[principal] == 0.0:
        raise ValueError("el rayo principal tiene potencia nula")
    resto = float(np.sum(p[:principal]) + np.sum(p[principal + 1:]))
    if resto == 0.0:
        return float("inf")
    return float(10.0 * np.log10(p[principal] / resto))


def ajustar(magnitudes, db=_VIGENTE):
    if db is _VIGENTE:
        db = nivel_actual()
    salida = np.array(magnitudes, copy=True)
    if salida.ndim != 1 or salida.size == 0:
        raise ValueError("magnitudes debe ser un array 1-D no vacío")
    if db is None:
        return salida
    db = float(db)
    if not np.isfinite(db):
        raise ValueError("db debe ser finito o None")
    dtype = salida.dtype if np.issubdtype(salida.dtype, np.inexact) else np.float64
    trabajo = salida.astype(np.complex128 if np.iscomplexobj(salida) else np.float64, copy=True)
    p = np.abs(trabajo) ** 2
    principal = int(np.argmax(p))
    resto = float(np.sum(p[:principal]) + np.sum(p[principal + 1:]))
    if resto == 0.0:
        return salida
    energia = float(np.sum(p))
    alfa = float(np.sqrt(p[principal] / (resto * 10.0 ** (db / 10.0))))
    trabajo[:principal] *= alfa
    trabajo[principal + 1:] *= alfa
    trabajo *= np.sqrt(energia / float(np.sum(np.abs(trabajo) ** 2)))
    return trabajo.astype(dtype, copy=False)
