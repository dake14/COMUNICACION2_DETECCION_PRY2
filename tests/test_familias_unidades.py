import numpy as np
import pytest

from src.banco import Config, sintetizar
from src.banco.modulaciones import Modulacion, catalogo_digital
from src.familias.conforme import ConjuntoConforme, salida_final
from src.familias.tasa import estimar_tasa
from src.familias.tonos import analizar_fsk

CATALOGO = catalogo_digital()
INVALIDAS = [np.zeros(0), np.ones(10), np.full(5000, np.nan + 0j), np.zeros(5000, complex)]


def senal(modulacion, sps, snr, semilla):
    c = Config(n_simbolos=2048, sps=sps, pulso="rrc", beta=0.35, span=10, snr_db=snr, cfo=0.0)
    return sintetizar(modulacion, "procedural", "ideal", c, np.random.default_rng(semilla))["iq"]


def fsk(nombre, orden, h):
    return Modulacion(nombre, "frecuencia", "cpm", orden=orden, h=(h, h))


@pytest.mark.parametrize("nombre", ["MSK", "2FSK", "4FSK", "QPSK"])
@pytest.mark.parametrize("sps", [4.0, 8.0, 16.0])
def test_la_tasa_ciega_sirve_para_envolvente_constante(nombre, sps):
    for semilla in range(3):
        r = estimar_tasa(senal(CATALOGO[nombre], sps, 20.0, semilla))
        assert abs(r["sps"] / sps - 1) < 0.02


@pytest.mark.parametrize("x", INVALIDAS)
def test_la_tasa_no_falla_con_entradas_invalidas(x):
    assert estimar_tasa(x)["sps"] is None


@pytest.mark.parametrize("modulacion, tonos, h, tolerancia", [
    (fsk("2FSK", 2, 1.0), 2, 1.0, 0.1),
    (fsk("4FSK", 4, 0.5), 4, 0.5, 0.075),
    (fsk("3FSK", 3, 0.8), 3, None, None),
    (CATALOGO["MSK"], 2, 0.5, 0.05),
])
def test_cuenta_los_tonos_y_el_indice(modulacion, tonos, h, tolerancia):
    for semilla in range(3):
        r = analizar_fsk(senal(modulacion, 16.0, 30.0, semilla), 16.0)
        assert r["tonos"] == tonos
        if h is not None:
            assert abs(r["h"] - h) < tolerancia


@pytest.mark.parametrize("x, sps", [(x, 8) for x in INVALIDAS] + [(np.ones(5000), 0), (np.ones(5000), np.nan)])
def test_los_tonos_no_fallan_con_entradas_invalidas(x, sps):
    assert analizar_fsk(x, sps)["tonos"] == 0


def datos_conformes(n, rng):
    y = rng.integers(0, 6, n)
    p = np.array([rng.dirichlet(1 + 6 * np.eye(6)[k]) for k in y])
    return p, y


@pytest.mark.parametrize("alfa", [0.05, 0.1, 0.2])
def test_la_cobertura_conforme_es_la_pedida(alfa):
    rng = np.random.default_rng(0)
    p, y = datos_conformes(5000, rng)
    pp, yy = datos_conformes(20000, rng)
    conjuntos = ConjuntoConforme(alfa).ajustar(p, y).conjunto(pp)
    cobertura = np.mean([k in c for k, c in zip(yy, conjuntos)])
    assert abs(cobertura - max(1 - alfa, np.mean(pp.argmax(1) == yy))) < 0.02


def test_la_salida_final_lista_equivalentes_y_alternativas():
    args = (np.array([0.3, 0.7]), ["BPSK", "QPSK"], [1, 0], {"QPSK": ("DQPSK",)})
    r = salida_final(*args, 0.6)
    assert r["modulacion"] == "QPSK" and r["misma_senal_que"] == ["DQPSK"]
    assert r["tambien_puede_ser"] == [{"nombre": "BPSK", "probabilidad": 0.3}]
    assert salida_final(*args, 0.8)["modulacion"] == "INDETERMINADA"


def test_el_conjunto_conforme_valida_sus_entradas():
    with pytest.raises(ValueError):
        ConjuntoConforme(1.5)
    with pytest.raises(RuntimeError):
        ConjuntoConforme().conjunto(np.ones(3) / 3)
    with pytest.raises(ValueError):
        ConjuntoConforme().ajustar(np.ones((2, 3)) / 3, np.array([0, 3]))
