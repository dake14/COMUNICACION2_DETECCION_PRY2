import numpy as np

from src.cascada import geometria as g
from src.cascada.autofoco import autofoco, entropia
from src.cascada.clasica import simbolos_clasicos, simbolos_sin_ecualizar
from src.cascada.evaluacion import evm_ciego
from src.cascada.sintesis import Condicion, transmitir


def _recibir(puntos, cond, semilla, n=1000):
    rng = np.random.default_rng(semilla)
    return transmitir(puntos, n, cond, rng)


def test_autofoco_16qam_con_giro():
    r = _recibir(g.qam(16), Condicion(snr_db=25.0, cfo=0.0012, tau=3.1, fase=1.0), 1)
    z, df, nitidez = autofoco(simbolos_sin_ecualizar(r["iq"]))
    assert abs(df - 8 * 0.0012) < 2e-3
    assert nitidez > 2
    assert evm_ciego(z, g.qam(16)) < 0.15


def test_autofoco_en_geometria_arbitraria():
    puntos = g.desconocidas_prueba()["ARB_A"]
    r = _recibir(puntos, Condicion(snr_db=28.0, cfo=-0.001, tau=1.0), 2)
    z, _, _ = autofoco(simbolos_sin_ecualizar(r["iq"]))
    assert evm_ciego(z, puntos) < 0.15


def test_entropia_baja_con_constelacion_nitida():
    rng = np.random.default_rng(3)
    puntos = g.psk(4)[rng.integers(0, 4, 800)]
    ruido = (rng.normal(size=800) + 1j * rng.normal(size=800)) * 0.7
    assert entropia(puntos) < entropia(puntos + ruido)


def test_cma_mejora_multitrayecto():
    taps = np.zeros(13, dtype=complex)
    taps[0], taps[12] = 1.0, 0.45j
    r = _recibir(g.psk(4), Condicion(snr_db=25.0, taps=taps, tau=2.0), 4, 1500)
    sin, _, _ = autofoco(simbolos_sin_ecualizar(r["iq"]))
    con, _, _ = autofoco(simbolos_clasicos(r["iq"]))
    assert evm_ciego(con[300:], g.psk(4)) < evm_ciego(sin[300:], g.psk(4))
