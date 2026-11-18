import numpy as np

from src.cascada import geometria as g
from src.cascada.clasica import tomar_simbolos
from src.cascada.sintesis import Condicion, condicion_aleatoria, transmitir


def test_catalogo_y_reservadas_tienen_potencia_unitaria():
    grupos = {**g.catalogo(), **g.desconocidas_validacion(), **g.desconocidas_prueba()}
    for nombre, z in grupos.items():
        assert np.isclose(np.mean(np.abs(z) ** 2), 1.0), nombre


def test_tamanos_de_las_constelaciones():
    assert {k: len(v) for k, v in g.catalogo().items()} == {
        "BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "64QAM": 64, "4PAM": 4}
    prueba = g.desconocidas_prueba()
    assert len(prueba["32QAM"]) == 32 and len(prueba["128QAM"]) == 128
    assert len(prueba["16APSK"]) == 16 and len(prueba["32APSK"]) == 32
    assert len(prueba["12PSK"]) == 12 and len(prueba["8QAM_ESTRELLA"]) == 8
    validacion = g.desconocidas_validacion()
    assert len(validacion["6PSK"]) == 6 and len(validacion["24QAM"]) == 24
    assert len(validacion["HEX19"]) == 19 and len(validacion["ANILLOS12"]) == 12


def test_distancia_invariante_a_fase():
    z = g.qam(16)
    assert g.distancia_geometrica(z, z * np.exp(0.7j)) < 0.03
    assert g.distancia_geometrica(z, g.psk(16)) > 0.05


def test_geometrias_aleatorias_no_copian_reservadas():
    rng = np.random.default_rng(1)
    for _ in range(40):
        z = g.geometria_aleatoria(rng)
        assert 2 <= len(z) <= 128
        assert not g.es_reservada(z)
    assert g.es_reservada(g.desconocidas_prueba()["32APSK"] * np.exp(0.3j))


def test_contar_anillos():
    assert g.contar_anillos(g.psk(8)) == 1
    assert g.contar_anillos(g.desconocidas_prueba()["32APSK"]) == 3


def test_snr_de_la_salida():
    limpio = transmitir(g.qam(16), 4000, Condicion(snr_db=200.0), np.random.default_rng(2))["iq"]
    ruidoso = transmitir(g.qam(16), 4000, Condicion(snr_db=5.0), np.random.default_rng(2))["iq"]
    assert np.isclose(np.mean(np.abs(limpio) ** 2), 1.0, atol=1e-3)
    assert np.isclose(np.mean(np.abs(ruidoso) ** 2), 1.0, atol=1e-3)
    ruido = ruidoso * np.sqrt(1 + 10 ** -0.5) - limpio
    assert abs(-10 * np.log10(np.mean(np.abs(ruido) ** 2)) - 5.0) < 0.5


def test_limpia_entrega_los_simbolos_en_sus_instantes():
    rng = np.random.default_rng(3)
    cond = Condicion(snr_db=200.0, tau=2.5, fase=0.4)
    r = transmitir(g.psk(4), 400, cond, rng)
    z = tomar_simbolos(r["limpia"])
    z = z[10:-10] / np.sqrt(np.mean(np.abs(z[10:-10]) ** 2))
    cuarta = np.abs(np.mean(z ** 4))
    assert cuarta > 0.97


def test_condicion_aleatoria_transmite_en_todos_los_pulsos():
    rng = np.random.default_rng(4)
    for _ in range(25):
        r = transmitir(g.geometria_aleatoria(rng), 128, condicion_aleatoria(rng), rng)
        assert len(r["iq"]) == len(r["limpia"])
        assert np.isfinite(r["iq"]).all()
