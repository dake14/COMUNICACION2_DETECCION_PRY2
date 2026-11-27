"""Fusión jerárquica del enrutador de familias con el decisor de constelación.

CONTRATO (unidad U10, cerrado):

Dependencias permitidas (lista cerrada): numpy.

- ``fusionar(p_familias, p_enrutador, p_decisor, familia_de, tonos=None, tonos_conocidos=(2, 4)) -> dict``
  - ``p_familias``: dict familia -> probabilidad (salida del enrutador). ``p_enrutador``: dict clase -> probabilidad
    (las clases del enrutador). ``p_decisor``: dict nombre -> probabilidad de ``Decisor.probabilidades`` (clases
    inscritas más ``"DESCONOCIDA"``), o ``None`` si la cascada no recuperó símbolos. ``familia_de``: dict clase ->
    familia. ``tonos``: int (conteo de ``analizar_fsk``) o ``None``.
  - ``ValueError`` si ``p_familias`` está vacío, si alguna de sus claves no es un valor de ``familia_de``, o si falta
    en ``p_enrutador`` alguna clase de ``familia_de``.
  - ``p_fondo = float(p_decisor.get("DESCONOCIDA", 0.0))`` si ``p_decisor`` no es None; si no, 0.0.
  - Para cada familia ``f`` de ``p_familias``, con ``clases`` = las claves de ``familia_de`` cuya familia es ``f``
    (en el orden de ``familia_de``):
    - Si ``f == "frecuencia"``: ``u = 1.0`` si ``tonos is not None and tonos >= 3 and tonos not in tonos_conocidos``,
      si no ``0.0``; ``q[c] = p_enrutador[c]``.
    - Si no: ``u = p_fondo``; ``q[c] = p_enrutador[c] * p_decisor.get(c, 0.0)`` si ``p_decisor`` no es None
      (producto de expertos). Si ``p_decisor`` es None o ``sum(q) <= 1e-12``: ``q[c] = p_enrutador[c]``.
      (Revisión: con el decisor solo, la exactitud dentro de la familia era 65,1 % frente a 74,9 % del enrutador;
      el producto da 78,3 % en la partición de calibración.)
    - ``s = sum(q)``; si ``s <= 1e-12``, ``q`` uniforme (``1.0`` por clase) y ``s = len(clases)``.
    - ``P[c] += p_familias[f] * (1 - u) * q[c] / s`` para cada clase y ``P["DESCONOCIDA"] += p_familias[f] * u``.
  - Devuelve ``{"probabilidades": P, "rama": rama}``. ``P``: dict con todas las clases de ``familia_de`` en su orden
    (0.0 las de familias ausentes de ``p_familias``) y ``"DESCONOCIDA"`` al final, todos ``float`` nativos.
    ``rama = "frecuencia"`` si la familia de mayor probabilidad en ``p_familias`` es ``"frecuencia"``, si no
    ``"constelacion"``.

- ``decidir(probabilidades, umbral, indeterminada="INDETERMINADA") -> dict``
  - ``probabilidades``: dict nombre -> probabilidad (no vacío; si no, ``ValueError``).
  - ``k`` = nombre de mayor probabilidad (empates: el primero en el orden del dict).
  - Devuelve ``{"modulacion": k si probabilidades[k] >= umbral si no indeterminada, "confianza": float(probabilidades[k])}``.

Criterio de aceptación:
1. Con ``familia_de = {"BPSK": "fase", "QPSK": "fase", "16QAM": "qam", "2FSK": "frecuencia", "4FSK": "frecuencia"}``,
   ``p_familias = {"fase": 0.6, "qam": 0.3, "frecuencia": 0.1}``, ``p_enrutador`` = 0.2 para cada clase,
   ``p_decisor = {"BPSK": 0.1, "QPSK": 0.3, "16QAM": 0.4, "DESCONOCIDA": 0.2}`` y ``tonos = 2``:
   P = BPSK 0.12, QPSK 0.36, 16QAM 0.24, 2FSK 0.05, 4FSK 0.05, DESCONOCIDA 0.18 (±1e-12); rama "constelacion";
   la suma es 1.
2. Igual pero ``tonos = 3``: 2FSK y 4FSK 0.0, DESCONOCIDA 0.28. Con ``tonos = 4``: igual que el punto 1.
3. ``p_decisor = None``: dentro de cada familia se reparte según ``p_enrutador`` (BPSK 0.3, QPSK 0.3, 16QAM 0.3) y
   DESCONOCIDA es 0.0 con ``tonos = 2``.
4. ``p_familias = {"frecuencia": 0.9, "fase": 0.1}`` da rama "frecuencia" y 16QAM 0.0.
5. ``decidir({"A": 0.7, "B": 0.3}, 0.6)`` da modulación "A", confianza 0.7; con umbral 0.8 da "INDETERMINADA".
6. Los errores descritos lanzan ``ValueError``.
7. Como el punto 1 pero con ``p_enrutador = {"BPSK": 0.6, "QPSK": 0.1, "16QAM": 0.1, "2FSK": 0.1, "4FSK": 0.1}``:
   BPSK 0.32, QPSK 0.16, 16QAM 0.24, 2FSK 0.05, 4FSK 0.05, DESCONOCIDA 0.18 (±1e-12).
"""
import numpy as np

DESCONOCIDA = "DESCONOCIDA"
FRECUENCIA = "frecuencia"


def _validar_fusion(p_familias, p_enrutador, familia_de):
    if not p_familias:
        raise ValueError("p_familias no puede estar vacío")
    familias_validas = set(familia_de.values())
    for familia in p_familias:
        if familia not in familias_validas:
            raise ValueError(f"Familia desconocida en p_familias: {familia!r}")
    for clase in familia_de:
        if clase not in p_enrutador:
            raise ValueError(f"Falta la clase {clase!r} en p_enrutador")


def _pesos_familia(familia, clases, p_enrutador, p_decisor, p_fondo, tonos, tonos_conocidos):
    if familia == FRECUENCIA:
        u = 1.0 if (tonos is not None and tonos >= 3 and tonos not in tonos_conocidos) else 0.0
        q = [float(p_enrutador[c]) for c in clases]
    else:
        u = p_fondo
        if p_decisor is not None:
            q = [float(p_enrutador[c]) * float(p_decisor.get(c, 0.0)) for c in clases]
            if sum(q) <= 1e-12:
                q = [float(p_enrutador[c]) for c in clases]
        else:
            q = [float(p_enrutador[c]) for c in clases]

    s = float(sum(q))
    if s <= 1e-12:
        q = [1.0] * len(clases)
        s = float(len(clases))
    return u, q, s


def fusionar(p_familias, p_enrutador, p_decisor, familia_de, tonos=None, tonos_conocidos=(2, 4)):
    _validar_fusion(p_familias, p_enrutador, familia_de)

    p_fondo = float(p_decisor.get(DESCONOCIDA, 0.0)) if p_decisor is not None else 0.0
    probabilidades = {clase: 0.0 for clase in familia_de}
    probabilidades[DESCONOCIDA] = 0.0

    for familia, p_fam in p_familias.items():
        clases = [clase for clase, fam in familia_de.items() if fam == familia]
        u, q, s = _pesos_familia(
            familia, clases, p_enrutador, p_decisor, p_fondo, tonos, tonos_conocidos
        )
        p_fam_f = float(p_fam)
        for clase, qc in zip(clases, q):
            probabilidades[clase] = float(probabilidades[clase] + p_fam_f * (1.0 - u) * qc / s)
        probabilidades[DESCONOCIDA] = float(probabilidades[DESCONOCIDA] + p_fam_f * u)

    familia_top = max(p_familias, key=lambda fam: float(p_familias[fam]))
    rama = FRECUENCIA if familia_top == FRECUENCIA else "constelacion"
    return {"probabilidades": probabilidades, "rama": rama}


def decidir(probabilidades, umbral, indeterminada="INDETERMINADA"):
    if not probabilidades:
        raise ValueError("probabilidades no puede estar vacío")
    k = max(probabilidades, key=lambda nombre: float(probabilidades[nombre]))
    confianza = float(probabilidades[k])
    return {
        "modulacion": k if confianza >= umbral else indeterminada,
        "confianza": confianza,
    }
