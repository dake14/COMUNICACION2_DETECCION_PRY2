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


FAMILIA_PRUEBA = {"BPSK": "fase", "QPSK": "fase", "16QAM": "qam", "2FSK": "frecuencia", "4FSK": "frecuencia"}
P_FAMILIAS = {"fase": 0.6, "qam": 0.3, "frecuencia": 0.1}
P_DECISOR = {"BPSK": 0.1, "QPSK": 0.3, "16QAM": 0.4, "DESCONOCIDA": 0.2}
PAREJO = {c: 0.2 for c in FAMILIA_PRUEBA}


def test_la_fusion_multiplica_familia_por_modulacion_dentro_de_ella():
    from src.familias.integracion import fusionar
    p = fusionar(P_FAMILIAS, PAREJO, P_DECISOR, FAMILIA_PRUEBA, 2)["probabilidades"]
    esperado = {"BPSK": 0.12, "QPSK": 0.36, "16QAM": 0.24, "2FSK": 0.05, "4FSK": 0.05, "DESCONOCIDA": 0.18}
    assert p == pytest.approx(esperado, abs=1e-12)
    sesgado = {"BPSK": 0.6, "QPSK": 0.1, "16QAM": 0.1, "2FSK": 0.1, "4FSK": 0.1}
    p = fusionar(P_FAMILIAS, sesgado, P_DECISOR, FAMILIA_PRUEBA, 2)["probabilidades"]
    assert p["BPSK"] == pytest.approx(0.32) and p["QPSK"] == pytest.approx(0.16)


def test_tres_tonos_vuelven_desconocida_la_rama_de_frecuencia():
    from src.familias.integracion import fusionar
    p = fusionar(P_FAMILIAS, PAREJO, P_DECISOR, FAMILIA_PRUEBA, 3)["probabilidades"]
    assert p["2FSK"] == 0 and p["DESCONOCIDA"] == pytest.approx(0.28)


def test_sin_constelacion_decide_el_enrutador():
    from src.familias.integracion import decidir, fusionar
    r = fusionar({"frecuencia": 0.9, "fase": 0.1}, PAREJO, None, FAMILIA_PRUEBA)
    assert r["rama"] == "frecuencia" and r["probabilidades"]["DESCONOCIDA"] == 0
    assert decidir({"A": 0.7, "B": 0.3}, 0.8)["modulacion"] == "INDETERMINADA"
    with pytest.raises(ValueError):
        fusionar({"otra": 1.0}, PAREJO, None, FAMILIA_PRUEBA)


def test_un_grupo_de_familias_comparte_el_especialista():
    from src.familias.integracion import fusionar
    p_esp = {"BPSK": 0.5, "QPSK": 0.25, "16QAM": 0.25}
    p = fusionar(P_FAMILIAS, PAREJO, P_DECISOR, FAMILIA_PRUEBA, 2, grupos=[("fase", "qam")],
                 p_especialista=p_esp)["probabilidades"]
    esperado = {"BPSK": 0.16, "QPSK": 0.24, "16QAM": 0.32, "2FSK": 0.05, "4FSK": 0.05, "DESCONOCIDA": 0.18}
    assert p == pytest.approx(esperado, abs=1e-12)
    for grupos in ([("fase", "frecuencia")], [("otra",)], [("fase",), ("fase", "qam")]):
        with pytest.raises(ValueError):
            fusionar(P_FAMILIAS, PAREJO, P_DECISOR, FAMILIA_PRUEBA, 2, grupos=grupos)


def _ruidosa(puntos, rng):
    s = puntos[rng.integers(0, len(puntos), 4000)] * np.exp(0.3j) / np.sqrt(np.mean(np.abs(puntos) ** 2))
    return s + (rng.normal(size=4000) + 1j * rng.normal(size=4000)) * np.sqrt(1e-3 / 2)


def test_las_vistas_de_constelacion_ven_cada_nivel():
    from scipy.signal import find_peaks
    from src.familias.detalle import vistas_constelacion
    rng = np.random.default_rng(0)
    picos = lambda c: len(find_peaks(c, height=0.2, distance=3)[0])
    qam = lambda niveles: np.array([a + 1j * b for a in niveles for b in niveles])
    v16 = vistas_constelacion(_ruidosa(qam([-3, -1, 1, 3]), rng))
    v64 = vistas_constelacion(_ruidosa(qam(range(-7, 8, 2)), rng))
    apsk = np.r_[np.exp(2j * np.pi * np.arange(4) / 4), 2.7 * np.exp(2j * np.pi * np.arange(12) / 12)]
    va = vistas_constelacion(_ruidosa(apsk, rng))
    assert picos(v16[0]) == picos(v16[1]) == 4 and picos(v64[0]) == picos(v64[1]) == 8 and picos(va[2]) == 2
    assert v16.dtype == np.float32 and v16.shape == (4, 256) and np.allclose(v16.max(1), 1)


@pytest.mark.parametrize("x", [None, np.zeros(0), np.ones(10), np.full(100, np.nan + 0j), np.zeros(100, complex)])
def test_las_vistas_no_fallan_con_entradas_invalidas(x):
    from src.familias.detalle import vistas_constelacion
    assert not vistas_constelacion(x).any()
