import json

import numpy as np
import pytest
from scipy.signal import welch

from experiments.armar_banco import armar
from src.banco import CANALES, GENERADORES, Config, sintetizar
from src.banco.generadores import Generador
from src.banco.modulaciones import (EQUIVALENTES, FAMILIAS, Modulacion, catalogo_digital, desconocidas_prueba_digital,
                                    desconocidas_validacion_digital, modulaciones_digitales, retardar)

CATALOGO = catalogo_digital()
TODAS = {k: v for k, (v, _, _) in modulaciones_digitales().items()}
PY_GEN = [k for k, v in GENERADORES.items() if not v.gnuradio]
PY_CAN = [k for k, v in CANALES.items() if not v.gnuradio]
PULSO = {"analogico": "nrz_butter"}


def cfg(generador="procedural", **cambios):
    base = dict(n_simbolos=2048, sps=8.0, pulso=PULSO.get(generador, "rrc"), beta=0.35, span=10, snr_db=200.0,
                cfo=0.0)
    return Config(**{**base, **cambios})


def iq(modulacion, generador="procedural", canal="ideal", semilla=1, **cambios):
    return sintetizar(modulacion, generador, canal, cfg(generador, **cambios), np.random.default_rng(semilla))


def test_catalogo_tiene_las_18_clases_y_desconocidas_separadas():
    assert set(CATALOGO) == {"OOK", "4ASK", "4PAM", "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", "16QAM", "32QAM",
                             "64QAM", "256QAM", "16APSK", "32APSK", "2FSK", "4FSK", "MSK", "GMSK"}
    desconocidas = {**desconocidas_validacion_digital(), **desconocidas_prueba_digital()}
    assert not set(desconocidas) & set(CATALOGO)
    assert {"3FSK", "8FSK"} <= set(desconocidas)
    assert all(m.familia in FAMILIAS for m in TODAS.values())
    assert set(EQUIVALENTES) <= set(CATALOGO)


@pytest.mark.parametrize("generador", PY_GEN)
@pytest.mark.parametrize("nombre", list(TODAS))
def test_cada_modulacion_produce_una_senal_valida(generador, nombre):
    r = iq(TODAS[nombre], generador, n_simbolos=128, sps=5.0, snr_db=20.0, cfo=0.001)
    assert len(r["iq"]) == 640 and np.isfinite(r["iq"]).all()
    json.dumps(r["parametros"])


@pytest.mark.parametrize("canal", PY_CAN)
@pytest.mark.parametrize("nombre", ["OQPSK", "PI4DQPSK", "4FSK", "GMSK", "OOK"])
def test_las_formas_nuevas_pasan_por_cada_canal(canal, nombre):
    r = iq(CATALOGO[nombre], "frecuencia", canal, n_simbolos=128, sps=5.5, snr_db=20.0)
    assert len(r["iq"]) == round(128 * 5.5) and np.isfinite(r["iq"]).all()


@pytest.mark.parametrize("nombre", ["2FSK", "GMSK", "OQPSK", "PI4DQPSK", "4ASK"])
def test_misma_semilla_misma_senal(nombre):
    a = iq(CATALOGO[nombre], "frecuencia", "tdl_rayleigh", semilla=7, n_simbolos=128)
    b = iq(CATALOGO[nombre], "frecuencia", "tdl_rayleigh", semilla=7, n_simbolos=128)
    assert np.array_equal(a["iq"], b["iq"]) and a["parametros"] == b["parametros"]


@pytest.mark.parametrize("generador", PY_GEN)
def test_ook_y_4ask_tienen_portadora_y_4pam_no(generador):
    portadora = {n: float(np.abs(np.mean(iq(CATALOGO[n], generador, semilla=3)["iq"])) ** 2)
                 for n in ("OOK", "4ASK", "4PAM")}
    assert portadora["OOK"] > 0.25 and portadora["4ASK"] > 0.35 and portadora["4PAM"] < 0.02


@pytest.mark.parametrize("generador", PY_GEN)
def test_2fsk_con_h_1_muestra_los_dos_tonos(generador):
    fsk = Modulacion("2FSK_H1", "frecuencia", "cpm", orden=2, h=(1.0, 1.0))
    f, p = welch(iq(fsk, generador, semilla=4)["iq"], nperseg=1024, return_onesided=False, detrend=False)
    for tono in (1 / 16, -1 / 16):
        k = int(np.argmin(np.abs(f - tono)))
        assert p[max(0, k - 2):k + 3].max() > 100 * np.median(p)


@pytest.mark.parametrize("generador", ["repo", "procedural", "frecuencia"])
def test_msk_tiene_envolvente_constante_y_frecuencia_en_un_cuarto(generador):
    x = iq(CATALOGO["MSK"], generador)["iq"].astype(np.complex128)
    envolvente = np.abs(x)
    assert envolvente.std() / envolvente.mean() < 0.01
    f = np.diff(np.unwrap(np.angle(x))) / (2 * np.pi)
    assert np.median(np.abs(f)) == pytest.approx(1 / 32, rel=0.02)


def test_gmsk_suaviza_la_frecuencia_y_sigue_de_envolvente_constante():
    msk = np.diff(np.unwrap(np.angle(iq(CATALOGO["MSK"], semilla=5)["iq"].astype(np.complex128))))
    gmsk = np.diff(np.unwrap(np.angle(iq(CATALOGO["GMSK"], semilla=5)["iq"].astype(np.complex128))))
    assert np.max(np.abs(np.diff(gmsk))) < 0.5 * np.max(np.abs(np.diff(msk)))
    assert np.std(np.abs(iq(CATALOGO["GMSK"], semilla=5)["iq"])) < 0.01


@pytest.mark.parametrize("generador", PY_GEN)
def test_oqpsk_no_pasa_cerca_de_cero_como_qpsk(generador):
    qpsk = Modulacion("Q", "fase", "lineal", CATALOGO["QPSK"].puntos)
    relativa = lambda z: np.abs(z) / np.sqrt(np.mean(np.abs(z) ** 2))
    oqpsk = np.quantile(relativa(iq(CATALOGO["OQPSK"], generador, semilla=2)["iq"]), 0.01)
    plana = np.quantile(relativa(iq(qpsk, generador, semilla=2)["iq"]), 0.01)
    assert oqpsk > 0.4 and oqpsk > 2.5 * plana


def test_oqpsk_nunca_usa_medio_seno_porque_seria_msk():
    r = iq(CATALOGO["OQPSK"], "procedural", pulso="semiseno", n_simbolos=64)
    assert r["parametros"]["modulacion"]["pulso"] == "rrc"


def test_pi4dqpsk_alterna_dos_cuaternas_de_fase(monkeypatch):
    monkeypatch.setitem(GENERADORES, "repo", Generador(lambda s, c, r: s, ("rrc",), False))
    s, _ = CATALOGO["PI4DQPSK"].transmitir("repo", cfg("repo"), 200, np.random.default_rng(6))
    resto = np.mod(np.angle(s), np.pi / 2)
    resto = np.minimum(resto, np.pi / 2 - resto)
    assert np.allclose(resto[0::2], np.pi / 4) and np.allclose(resto[1::2], 0, atol=1e-9)


def test_retardo_fraccionario_desplaza_medio_simbolo():
    t = np.arange(512)
    x = np.exp(-((t - 200) / 20.0) ** 2)
    y = retardar(x, 4.5).real
    assert np.argmax(y) in (204, 205) and np.sum(y * t) / np.sum(y) == pytest.approx(204.5, abs=0.05)


def test_parametros_de_fsk_quedan_registrados_y_en_su_rango():
    for semilla in range(20):
        m = iq(CATALOGO["4FSK"], semilla=semilla, n_simbolos=64)["parametros"]["modulacion"]
        assert 0.25 <= m["h"] <= 1.0 and m["bt"] is None
        g = iq(CATALOGO["GMSK"], semilla=semilla, n_simbolos=64)["parametros"]["modulacion"]
        assert g["h"] == 0.5 and g["bt"] in (0.25, 0.3, 0.5)


def test_armar_con_catalogo_digital_registra_familia(tmp_path):
    filas = armar(tmp_path, ["procedural"], ["ideal", "nakagami"], ["GMSK", "OQPSK", "8FSK", "3FSK"], por_celda=1,
                  semilla=9, trabajos=1, n_simbolos=(64,), catalogo="digital")
    assert len(filas) == 8
    assert {f["etiqueta"] for f in filas} == {"GMSK", "OQPSK", "DESCONOCIDA"}
    assert {f["grupo"] for f in filas} == {"catalogo", "prueba", "validacion"}
    assert {f["familia"] for f in filas} == {"frecuencia", "fase"}
    for f in filas:
        assert np.isfinite(np.fromfile(tmp_path / f["archivo"], np.complex64)).all()
