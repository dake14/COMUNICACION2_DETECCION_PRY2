import json

import numpy as np
import pytest

from experiments.armar_banco import armar
from src.banco import CANALES, GENERADORES, Config, muestrear_config, sintetizar
from src.banco.canales import _retrasar
from src.banco.canales_extra import CANALES_CON_ECOS, TODOS_CANALES
from src.banco.isi import nivel_isi
from src.banco.verificacion import ancho_ocupado, prominencia_ciclica, snr_medida, tasa_ciclica
from src.cascada import geometria as g

QPSK = g.catalogo()["QPSK"]
PY_GEN = [k for k, v in GENERADORES.items() if not v.gnuradio]
PY_CAN = [k for k, v in CANALES.items() if not v.gnuradio]
PULSO = {"repo": "rrc", "procedural": "rrc", "frecuencia": "rrc", "analogico": "nrz_butter"}


def cfg(generador="procedural", **cambios):
    base = dict(n_simbolos=64, sps=4.0, pulso=PULSO.get(generador, "rrc"), beta=0.35, span=10, snr_db=20.0, cfo=0.001)
    return Config(**{**base, **cambios})


def test_el_banco_tiene_variedad():
    assert len(GENERADORES) >= 6 and len(PY_GEN) >= 4
    assert len(CANALES) >= 18 and len(PY_CAN) >= 14
    assert sum(v.gnuradio for v in GENERADORES.values()) >= 2


@pytest.mark.parametrize("generador", PY_GEN)
@pytest.mark.parametrize("canal", PY_CAN)
def test_cada_cruce_produce_una_senal_valida(generador, canal):
    r = sintetizar(QPSK, generador, canal, cfg(generador), np.random.default_rng(1))
    assert len(r["iq"]) == 256
    assert np.isfinite(r["iq"]).all() and np.mean(np.abs(r["iq"]) ** 2) > 0
    json.dumps(r["parametros"])


@pytest.mark.parametrize("canal", [c for c in PY_CAN if c != "ideal"])
def test_cada_canal_altera_la_senal(canal):
    c = cfg(snr_db=200.0, cfo=0.0, n_simbolos=128)
    a = sintetizar(QPSK, "procedural", "ideal", c, np.random.default_rng(2))["iq"]
    b = sintetizar(QPSK, "procedural", canal, c, np.random.default_rng(2))["iq"]
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    assert abs(np.vdot(a, b)) < 0.999999


def test_misma_semilla_misma_senal():
    a = sintetizar(QPSK, "frecuencia", "tdl_rayleigh", cfg("frecuencia"), np.random.default_rng(7))
    b = sintetizar(QPSK, "frecuencia", "tdl_rayleigh", cfg("frecuencia"), np.random.default_rng(7))
    assert np.array_equal(a["iq"], b["iq"]) and a["parametros"] == b["parametros"]


@pytest.mark.parametrize("snr", [0.0, 10.0, 25.0])
def test_snr_pedida(snr):
    r = sintetizar(QPSK, "procedural", "ideal", cfg(n_simbolos=4096, snr_db=snr), np.random.default_rng(3))
    assert abs(snr_medida(r) - snr) < 0.25


@pytest.mark.parametrize("generador", PY_GEN)
def test_la_tasa_de_simbolo_es_la_pedida(generador):
    c = cfg(generador, n_simbolos=2048, sps=6.0, snr_db=60.0, cfo=0.0)
    r = sintetizar(QPSK, generador, "ideal", c, np.random.default_rng(4))
    assert tasa_ciclica(r["iq"]) == pytest.approx(1 / 6, rel=0.01)


@pytest.mark.parametrize("generador", [g_ for g_ in PY_GEN if GENERADORES[g_].fraccionario])
def test_tasa_fraccionaria(generador):
    c = cfg(generador, n_simbolos=2048, sps=5.5, snr_db=60.0, cfo=0.0)
    r = sintetizar(QPSK, generador, "ideal", c, np.random.default_rng(5))
    assert len(r["iq"]) == round(2048 * 5.5)
    assert tasa_ciclica(r["iq"]) == pytest.approx(1 / 5.5, rel=0.01)


def test_tasa_del_pulso_de_medio_seno():
    c = cfg(n_simbolos=2048, sps=6.0, pulso="semiseno", snr_db=60.0, cfo=0.0)
    r = sintetizar(QPSK, "procedural", "ideal", c, np.random.default_rng(10))
    assert prominencia_ciclica(r["iq"], 1 / 6, derivada=True) > 10


@pytest.mark.parametrize("generador", ["repo", "procedural", "frecuencia"])
def test_ancho_de_banda_rrc(generador):
    c = cfg(generador, n_simbolos=2048, sps=8.0, beta=0.35, snr_db=60.0, cfo=0.0)
    r = sintetizar(QPSK, generador, "ideal", c, np.random.default_rng(6))
    assert ancho_ocupado(r["iq"]) == pytest.approx(1.35 / 8, rel=0.15)


@pytest.mark.parametrize("generador", list(GENERADORES))
def test_configuracion_respeta_lo_que_admite_el_generador(generador):
    rng = np.random.default_rng(8)
    info = GENERADORES[generador]
    for _ in range(100):
        c = muestrear_config(generador, rng)
        assert c.pulso in info.pulsos
        assert info.fraccionario or float(c.sps).is_integer()
        assert c.span % 2 == 0


def test_armar_cruza_todo_y_registra(tmp_path):
    filas = armar(tmp_path, ["procedural", "repo"], ["ideal", "mp_corto", "cuantizacion"], ["QPSK", "12PSK"],
                  por_celda=2, semilla=9, trabajos=1, n_simbolos=(64,))
    guardadas = [json.loads(x) for x in (tmp_path / "manifiesto.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(filas) == len(guardadas) == 2 * 3 * 2 * 2
    assert {f["etiqueta"] for f in guardadas} == {"QPSK", "DESCONOCIDA"}
    for f in guardadas:
        iq = np.fromfile(tmp_path / f["archivo"], np.complex64)
        assert len(iq) == f["muestras"] and np.isfinite(iq).all()
    assert len({f["archivo"] for f in guardadas}) == len(guardadas)


# ───────────────────────────  Regulador de ISI  ──────────────────────
def _senal_isi(n=4096, sps=4):
    rng = np.random.default_rng(3)
    u = np.zeros(n * sps, complex)
    u[::sps] = (rng.choice([-1, 1], n) + 1j * rng.choice([-1, 1], n)) / np.sqrt(2)
    return np.convolve(u, np.ones(sps) / np.sqrt(sps), "same"), sps


def _sir_db(x, y, bloque=256):
    """Señal / ISI con una ganancia por bloque y el mejor retardo del trayecto principal (puede no llegar primero)."""
    mejor = -np.inf
    for d in np.arange(0, 12, 0.25):
        xd = _retrasar(x, d)
        sig = err = 0.0
        for a in range(0, len(x) - bloque + 1, bloque):
            xb, yb = xd[a:a + bloque], y[a:a + bloque]
            h = np.vdot(xb, yb) / np.vdot(xb, xb)
            sig += np.sum(np.abs(h * xb) ** 2)
            err += np.sum(np.abs(yb - h * xb) ** 2)
        mejor = max(mejor, 10 * np.log10(sig / err))
    return mejor


@pytest.mark.parametrize("canal", [c for c in CANALES_CON_ECOS if not TODOS_CANALES[c].gnuradio])
def test_el_regulador_suaviza_los_ecos(canal):
    x, sps = _senal_isi()
    sir = {}
    for db in (0.0, 30.0):
        with nivel_isi(db):
            y, _ = TODOS_CANALES[canal].antes(x, sps, np.random.default_rng(5))
        sir[db] = _sir_db(x, np.asarray(y)[:len(x)])
    assert sir[30.0] > sir[0.0] + 5, sir


def test_sin_regulador_el_canal_no_cambia():
    x, sps = _senal_isi()
    a, _ = TODOS_CANALES["mp_largo"].antes(x, sps, np.random.default_rng(5))
    with nivel_isi(None):
        b, _ = TODOS_CANALES["mp_largo"].antes(x, sps, np.random.default_rng(5))
    assert np.array_equal(a, b)
    assert cfg().isi_db is None


def test_armar_con_regulador_solo_toca_los_ecos(tmp_path):
    comun = dict(generadores=["procedural", "repo"], canales=["ideal", "mp_largo", "dos_rayos"], nombres=["QPSK"],
                 por_celda=6, semilla=9, trabajos=1, n_simbolos=(64,))
    base = {f["archivo"]: f for f in armar(tmp_path / "base", **comun)}
    regulado = armar(tmp_path / "isi", **comun, isi=("aleatorio", 0.5, 0.0, 30.0))
    niveles = [f["isi_db"] for f in regulado if f["canal"] != "ideal"]
    assert any(v is None for v in niveles) and any(v is not None and 0 <= v <= 30 for v in niveles)
    for f in regulado:
        igual = np.array_equal(np.fromfile(tmp_path / "isi" / f["archivo"], np.complex64),
                               np.fromfile(tmp_path / "base" / f["archivo"], np.complex64))
        assert igual == (f["isi_db"] is None), f["archivo"]
        assert f["snr_db"] == base[f["archivo"]]["snr_db"] and f["sps"] == base[f["archivo"]]["sps"]
    ciclo = armar(tmp_path / "niveles", **comun, isi=("niveles", (0.0, 10.0, 20.0)))
    assert {f["isi_db"] for f in ciclo if f["canal"] == "mp_largo"} == {0.0, 10.0, 20.0}
    assert {f["isi_db"] for f in ciclo if f["canal"] == "ideal"} == {None}
