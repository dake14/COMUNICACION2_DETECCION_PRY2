"""Fusión jerárquica del enrutador de familias con el decisor de constelación.

CONTRATO (unidad U10, cerrado):

Dependencias permitidas (lista cerrada): numpy.

- ``fusionar(p_familias, p_enrutador, p_decisor, familia_de, tonos=None, tonos_conocidos=(2, 4), grupos=None,
  p_especialista=None, consistentes=None) -> dict``
  - ``p_familias``: dict familia -> probabilidad (salida del enrutador). ``p_enrutador``: dict clase -> probabilidad
    (las clases del enrutador). ``p_decisor``: dict nombre -> probabilidad de ``Decisor.probabilidades`` (clases
    inscritas más ``"DESCONOCIDA"``), o ``None`` si la cascada no recuperó símbolos. ``familia_de``: dict clase ->
    familia. ``tonos``: int (conteo de ``analizar_fsk``) o ``None``. ``grupos``: lista de tuplas de familias que se
    tratan como un solo grupo (revisión: el especialista QAM+APSK decide también entre esas dos familias), o ``None``.
    ``p_especialista``: dict clase -> probabilidad del especialista de familia (solo las clases que cubre), o ``None``.
    ``consistentes`` (revisión U10d): dict clase -> peso ``w`` en [0, 1] (1 = los símbolos caben en la constelación
    de esa clase; ver ``src/familias/ajuste.py``), o ``None`` (todas 1). Una clase ausente del dict pesa 1.
  - Grupos efectivos: los de ``grupos`` en su orden, y luego cada familia de ``p_familias`` que no esté en ninguno,
    como grupo de una sola familia (en el orden de ``p_familias``). Una familia de un grupo que no esté en
    ``p_familias`` cuenta con probabilidad 0.
  - ``ValueError`` si ``p_familias`` está vacío, si alguna de sus claves no es un valor de ``familia_de``, o si falta
    en ``p_enrutador`` alguna clase de ``familia_de``. También si en ``grupos`` hay una familia que no es un valor de
    ``familia_de``, una familia repetida (en el mismo grupo o en dos), o un grupo de más de una familia que incluye
    ``"frecuencia"``. También si algún peso de ``consistentes`` no es finito o está fuera de [0, 1].
  - ``p_fondo = float(p_decisor.get("DESCONOCIDA", 0.0))`` si ``p_decisor`` no es None; si no, 0.0.
  - Para cada grupo efectivo ``g``, con ``p_g`` = suma de ``p_familias.get(f, 0.0)`` para ``f`` en ``g`` y
    ``clases`` = las claves de ``familia_de`` cuya familia está en ``g`` (en el orden de ``familia_de``):
    - Si ``g == ("frecuencia",)``: ``u = 1.0`` si ``tonos is not None and tonos >= 3 and tonos not in tonos_conocidos``,
      si no ``0.0``; ``q[c] = p_enrutador[c]``.
    - Si no: ``u = p_fondo``; ``q[c] = p_enrutador[c] * p_decisor.get(c, 0.0)`` si ``p_decisor`` no es None
      (producto de expertos). Si ``p_decisor`` es None o ``sum(q) <= 1e-12``: ``q[c] = p_enrutador[c]``. Después, si
      ``p_especialista`` no es None, ``q[c] *= p_especialista.get(c, 1.0)``; si eso deja ``sum(q) <= 1e-12``, se
      vuelve al ``q`` de antes de multiplicar.
      (Revisión: con el decisor solo, la exactitud dentro de la familia era 65,1 % frente a 74,9 % del enrutador;
      el producto da 78,3 % en la partición de calibración.)
    - ``s = sum(q)``; si ``s <= 1e-12``, ``q`` uniforme (``1.0`` por clase) y ``s = len(clases)``.
    - ``P[c] += p_g * (1 - u) * q[c] / s`` para cada clase y ``P["DESCONOCIDA"] += p_g * u``.
    - Revisión U10d: en los grupos que no son ``("frecuencia",)``, con ``w[c] = consistentes.get(c, 1.0)``, el
      término de cada clase pasa a ``p_g * (1 - u) * q[c] * w[c] / s`` y la masa que pierde,
      ``p_g * (1 - u) * q[c] * (1 - w[c]) / s``, se suma a ``P["DESCONOCIDA"]``. En el grupo de frecuencia
      ``consistentes`` no se usa.
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
8. Como el punto 1 con ``grupos = [("fase", "qam")]`` y ``p_especialista = {"BPSK": 0.5, "QPSK": 0.25, "16QAM": 0.25}``:
   BPSK 0.16, QPSK 0.24, 16QAM 0.32, 2FSK 0.05, 4FSK 0.05, DESCONOCIDA 0.18 (±1e-12); la suma es 1.
9. Como el punto 1 con ese ``p_especialista`` y sin ``grupos``: BPSK 0.192, QPSK 0.288, 16QAM 0.24 (±1e-12).
10. ``grupos`` con ``("fase", "frecuencia")``, con ``("otra",)`` o con ``[("fase",), ("fase", "qam")]`` lanzan
    ``ValueError``. ``grupos = [("frecuencia",)]`` da lo mismo que el punto 1.
11. Como el punto 1 con ``consistentes = {"QPSK": 0.0, "2FSK": 0.0}``: BPSK 0.12, QPSK 0.0, 16QAM 0.24, 2FSK 0.05,
    4FSK 0.05, DESCONOCIDA 0.54 (±1e-12); la suma es 1. Con ``{"QPSK": 0.5}``: QPSK 0.18, DESCONOCIDA 0.36.
    ``consistentes = {}`` o con todos los pesos 1 da lo mismo que el punto 1.
12. ``consistentes = {"BPSK": 1.5}``, ``{"BPSK": -0.1}`` o ``{"BPSK": float("nan")}`` lanzan ``ValueError``.
"""
import numpy as np

DESCONOCIDA = "DESCONOCIDA"
FRECUENCIA = "frecuencia"


def _validar_fusion(p_familias, p_enrutador, familia_de, grupos=None, consistentes=None):
    if not p_familias:
        raise ValueError("p_familias no puede estar vacío")
    familias_validas = set(familia_de.values())
    for familia in p_familias:
        if familia not in familias_validas:
            raise ValueError(f"Familia desconocida en p_familias: {familia!r}")
    for clase in familia_de:
        if clase not in p_enrutador:
            raise ValueError(f"Falta la clase {clase!r} en p_enrutador")

    if grupos is not None:
        vistas = set()
        for grupo in grupos:
            if len(grupo) > 1 and FRECUENCIA in grupo:
                raise ValueError(
                    f"Un grupo de más de una familia no puede incluir {FRECUENCIA!r}: {grupo!r}"
                )
            for familia in grupo:
                if familia not in familias_validas:
                    raise ValueError(f"Familia desconocida en grupos: {familia!r}")
                if familia in vistas:
                    raise ValueError(f"Familia repetida en grupos: {familia!r}")
                vistas.add(familia)

    if consistentes is not None:
        for clase, peso in consistentes.items():
            try:
                w = float(peso)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Peso inválido en consistentes para {clase!r}: {peso!r}") from exc
            if not np.isfinite(w) or w < 0.0 or w > 1.0:
                raise ValueError(f"Peso fuera de [0, 1] o no finito en consistentes para {clase!r}: {peso!r}")


def _grupos_efectivos(grupos, p_familias):
    efectivos = [tuple(g) for g in grupos] if grupos is not None else []
    en_grupos = {familia for grupo in efectivos for familia in grupo}
    for familia in p_familias:
        if familia not in en_grupos:
            efectivos.append((familia,))
    return efectivos


def _pesos_grupo(
    grupo,
    clases,
    p_enrutador,
    p_decisor,
    p_fondo,
    tonos,
    tonos_conocidos,
    p_especialista,
):
    if grupo == (FRECUENCIA,):
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

        if p_especialista is not None:
            q_esp = [qc * float(p_especialista.get(c, 1.0)) for c, qc in zip(clases, q)]
            if sum(q_esp) > 1e-12:
                q = q_esp

    s = float(sum(q))
    if s <= 1e-12:
        q = [1.0] * len(clases)
        s = float(len(clases))
    return u, q, s


def fusionar(p_familias, p_enrutador, p_decisor, familia_de, tonos=None, tonos_conocidos=(2, 4), grupos=None,
             p_especialista=None, consistentes=None):
    _validar_fusion(p_familias, p_enrutador, familia_de, grupos=grupos, consistentes=consistentes)

    p_fondo = float(p_decisor.get(DESCONOCIDA, 0.0)) if p_decisor is not None else 0.0
    probabilidades = {clase: 0.0 for clase in familia_de}
    probabilidades[DESCONOCIDA] = 0.0

    for grupo in _grupos_efectivos(grupos, p_familias):
        grupo_set = set(grupo)
        clases = [clase for clase, fam in familia_de.items() if fam in grupo_set]
        if not clases:
            continue
        p_g = sum(float(p_familias.get(fam, 0.0)) for fam in grupo)
        u, q, s = _pesos_grupo(
            grupo,
            clases,
            p_enrutador,
            p_decisor,
            p_fondo,
            tonos,
            tonos_conocidos,
            p_especialista,
        )
        es_frecuencia = grupo == (FRECUENCIA,)
        for clase, qc in zip(clases, q):
            w = 1.0 if (es_frecuencia or consistentes is None) else float(consistentes.get(clase, 1.0))
            probabilidades[clase] = float(probabilidades[clase] + p_g * (1.0 - u) * qc * w / s)
            probabilidades[DESCONOCIDA] = float(probabilidades[DESCONOCIDA] + p_g * (1.0 - u) * qc * (1.0 - w) / s)
        probabilidades[DESCONOCIDA] = float(probabilidades[DESCONOCIDA] + p_g * u)

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
