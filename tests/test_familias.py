import numpy as np
import pytest

from src.banco import Config, sintetizar
from src.banco.modulaciones import Modulacion, catalogo_digital
from src.familias.enrutador import Enrutador, Fusion
from src.familias.especialistas import Especialista
from src.familias.vistas import FORMAS, VISTAS, estimar_banda, normalizar_banda, vistas

CATALOGO = catalogo_digital()


def senal(nombre, sps=8.0, snr=40.0, cfo=0.0, n=2048, semilla=1, modulacion=None):
    c = Config(n_simbolos=n, sps=sps, pulso="rrc", beta=0.35, span=10, snr_db=snr, cfo=cfo)
    return sintetizar(modulacion or CATALOGO[nombre], "procedural", "ideal", c, np.random.default_rng(semilla))["iq"]


@pytest.mark.parametrize("sps", [3.0, 8.0, 24.0])
def test_el_ancho_estimado_sigue_a_la_tasa(sps):
    _, ancho = estimar_banda(senal("QPSK", sps))
    assert 1.0 < ancho * sps < 1.5


def test_el_centro_estimado_es_el_desvio_de_portadora():
    centro, _ = estimar_banda(senal("16QAM", 8.0, cfo=0.03))
    assert centro == pytest.approx(0.03, abs=0.004)


@pytest.mark.parametrize("sps", [3.0, 16.0])
def test_la_banda_normalizada_ocupa_la_mitad(sps):
    x, info = normalizar_banda(senal("QPSK", sps, cfo=0.01))
    centro, ancho = estimar_banda(x)
    assert ancho == pytest.approx(0.5, rel=0.12) and abs(centro) < 0.02
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("largo", [40, 700, 70000])
def test_las_vistas_tienen_forma_fija_y_son_finitas(largo):
    rng = np.random.default_rng(2)
    v, _ = vistas(rng.normal(size=largo) + 1j * rng.normal(size=largo))
    for k in VISTAS:
        assert v[k].shape == FORMAS[k] and np.isfinite(v[k]).all()


def test_las_vistas_casi_no_dependen_de_la_tasa():
    a, _ = vistas(senal("8PSK", 4.0))
    b, _ = vistas(senal("8PSK", 16.0, semilla=3))
    for k in ("espectro", "envolvente"):
        u, w = a[k].ravel(), b[k].ravel()
        assert np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)) > 0.9


def _linea(fila):
    return float(fila.max() - np.median(fila))


def test_las_potencias_separan_bpsk_qpsk_y_8psk():
    bpsk, qpsk, psk8 = (vistas(senal(n))[0]["potencias"] for n in ("BPSK", "QPSK", "8PSK"))
    assert _linea(bpsk[0]) > _linea(qpsk[0]) + 1.0
    assert _linea(qpsk[1]) > _linea(qpsk[0]) + 0.5
    assert _linea(psk8[2]) > _linea(psk8[1]) + 0.5


def test_la_frecuencia_instantanea_de_2fsk_tiene_dos_niveles():
    fsk = Modulacion("2FSK_H1", "frecuencia", "cpm", orden=2, h=(1.0, 1.0))
    h = vistas(senal("", modulacion=fsk))[0]["frecuencia"][1]
    picos = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.5]
    assert len(picos) == 2 and (picos[0] - 63.5) * (picos[1] - 63.5) < 0


def test_la_envolvente_distingue_constante_de_anillos():
    msk = vistas(senal("MSK"))[0]["envolvente"][0]
    qam = vistas(senal("64QAM"))[0]["envolvente"][0]
    assert np.sum(msk > 0.2) < np.sum(qam > 0.2)


def test_el_enrutador_se_guarda_carga_y_reparte_probabilidad(tmp_path):
    clases = ["BPSK", "QPSK", "2FSK"]
    familias = {"BPSK": "fase", "QPSK": "fase", "2FSK": "frecuencia"}
    especialistas = {k: Especialista(FORMAS[k][0], len(clases)).eval() for k in VISTAS}
    rng = np.random.default_rng(4)
    logp = {k: np.log(rng.dirichlet(np.ones(3), 60)) for k in VISTAS}
    fusion = Fusion(clases, familias).ajustar(logp, np.arange(60) % 3)
    Enrutador(especialistas, fusion).guardar(tmp_path)
    r = Enrutador.cargar(tmp_path).analizar(senal("QPSK", n=256))
    assert sum(r["clases"].values()) == pytest.approx(1.0)
    assert r["familias"]["fase"] == pytest.approx(r["clases"]["BPSK"] + r["clases"]["QPSK"])
    assert set(r["especialistas"]) == set(VISTAS)
