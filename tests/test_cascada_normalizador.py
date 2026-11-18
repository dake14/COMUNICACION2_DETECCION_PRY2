import numpy as np
import pytest

from src.cascada import geometria as g
from src.cascada.normalizador import normalizar
from src.cascada.sintesis import Condicion, transmitir


@pytest.mark.parametrize("sps", [4.0, 5.3, 8.0, 13.0])
@pytest.mark.parametrize("nombre", ["QPSK", "16QAM"])
def test_recupera_tasa(sps, nombre):
    rng = np.random.default_rng(int(sps * 10))
    cond = Condicion(snr_db=15.0, beta=0.35, tau=1.3, fase=0.2)
    iq = transmitir(g.catalogo()[nombre], 1500, cond, rng, sps_salida=sps)["iq"]
    h = normalizar(iq)[0]
    assert abs(h.sps_origen - sps) / sps < 0.01


def test_corrige_cfo_grueso():
    rng = np.random.default_rng(5)
    cond = Condicion(snr_db=15.0, cfo=0.03)
    iq = transmitir(g.qam(16), 1500, cond, rng)["iq"]
    assert abs(normalizar(iq)[0].cfo_grueso - 0.03) < 0.004


def test_invariante_a_escala():
    rng = np.random.default_rng(6)
    iq = transmitir(g.psk(8), 800, Condicion(snr_db=15.0), rng)["iq"]
    a, b = normalizar(iq)[0], normalizar(iq * 100)[0]
    assert a.sps_origen == b.sps_origen
    assert np.allclose(a.iq, b.iq, atol=1e-4)


@pytest.mark.parametrize("semilla", range(12))
def test_ruido_puro_no_rompe_el_normalizador(semilla):
    rng = np.random.default_rng(semilla)
    assert normalizar(rng.normal(size=8192) + 1j * rng.normal(size=8192))


def test_tasa_forzada():
    rng = np.random.default_rng(7)
    iq = transmitir(g.psk(4), 400, Condicion(snr_db=10.0), rng, sps_salida=4)["iq"]
    h = normalizar(iq, sps=4)
    assert len(h) == 1 and h[0].sps_origen == 4.0
    assert abs(len(h[0].iq) - 2 * len(iq)) <= 2
