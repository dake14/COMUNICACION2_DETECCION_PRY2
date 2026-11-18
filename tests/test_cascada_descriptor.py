import numpy as np
import torch

from src.cascada import geometria as g
from src.cascada.decisor import DESCONOCIDA, INDETERMINADA, Decisor
from src.cascada.descriptor import (Descriptor, aumentar, calcular_huellas, cargar_descriptor,
                                    guardar_descriptor, imagen_densidad, perdida_contrastiva)
from src.cascada.detector import detectar_ciego, snr_ciega_db
from src.cascada.sintesis import Condicion, transmitir


def test_imagen_acotada_e_invariante_a_escala():
    rng = np.random.default_rng(0)
    z = g.qam(16)[rng.integers(0, 16, 500)]
    a = imagen_densidad(z)
    assert a.shape == (2, 64, 64) and a.min() >= 0 and a.max() <= 1.5
    assert np.allclose(a, imagen_densidad(z * 37.0))


def test_huella_unitaria_y_red_pequena(tmp_path):
    m = Descriptor()
    assert sum(p.numel() for p in m.parameters()) < 300_000
    x = np.stack([imagen_densidad(g.psk(4)), imagen_densidad(g.qam(16))])
    z, geo = calcular_huellas(m, x)
    assert z.shape == (2, 128) and geo.shape == (2, 2)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)
    ruta = tmp_path / "d.pt"
    guardar_descriptor(m, ruta)
    assert np.allclose(z, calcular_huellas(cargar_descriptor(ruta), x)[0])


def test_aumento_conserva_forma():
    x = torch.rand(4, 2, 64, 64)
    assert aumentar(x).shape == x.shape


def test_perdida_contrastiva_premia_pares_iguales():
    z = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)
    otro = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)
    assert perdida_contrastiva(z, z) < perdida_contrastiva(z, otro)


def _nube(rng, centro, n=300):
    return np.asarray(centro) + 0.2 * rng.normal(size=(n, 6))


def _decisor(rng):
    c = {"A": np.eye(6)[0] * 3, "B": np.eye(6)[1] * 3}
    fondo = rng.uniform(-5, 5, size=(800, 6))
    d = Decisor(dim_max=6, componentes=1, componentes_fondo=4).ajustar(fondo, {k: _nube(rng, v) for k, v in c.items()})
    return d, c


def test_decisor_reconoce_y_rechaza():
    rng = np.random.default_rng(1)
    d, c = _decisor(rng)
    raro = np.array([[-4, 4, -4, 4, -4, 4.0]])
    z = np.vstack([_nube(rng, c["A"], 50), _nube(rng, c["B"], 50), raro + 0.3 * rng.normal(size=(50, 6))])
    d.calibrar(z, ["A"] * 50 + ["B"] * 50 + [DESCONOCIDA] * 50)
    assert d.decidir(_nube(rng, c["A"], 1))["modulacion"] == "A"
    assert d.decidir(raro)["modulacion"] in (DESCONOCIDA, INDETERMINADA)
    assert d.decidir(raro)["modulacion"] != "A"
    assert np.isclose(sum(d.decidir(raro)["probabilidades"].values()), 1.0)


def test_inscribir_sin_reajustar_pca():
    rng = np.random.default_rng(2)
    d, _ = _decisor(rng)
    componentes = d.pca.components_.copy()
    nueva = np.array([-3.0, 3.0, 0, 0, 0, 0])
    d.inscribir("C", _nube(rng, nueva))
    assert "C" in d.nombres and np.array_equal(componentes, d.pca.components_)
    d.umbral = 0.3
    assert d.decidir(_nube(rng, nueva, 1))["modulacion"] == "C"


def test_inscribir_despues_de_calibrar():
    rng = np.random.default_rng(4)
    d, c = _decisor(rng)
    raro = np.array([-4, 4, -4, 4, -4, 4.0])
    z = np.vstack([_nube(rng, c["A"], 50), _nube(rng, c["B"], 50), _nube(rng, raro, 50)])
    calidad = rng.normal(size=(150, 3))
    d.calibrar(z, ["A"] * 50 + ["B"] * 50 + [DESCONOCIDA] * 50, calidad)
    nueva = np.array([0, 0, 3.0, 0, 0, 0])
    d.inscribir("C", _nube(rng, nueva))
    p = d.probabilidades(_nube(rng, nueva, 1), calidad[:1])
    assert p.shape == (1, 4) and np.isclose(p.sum(), 1.0)
    assert p[0, :-1].argmax() == d.nombres.index("C")


def test_detector_ciego_y_snr():
    rng = np.random.default_rng(3)
    ruido = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    assert not detectar_ciego(ruido)["detectada"]
    iq = transmitir(g.qam(16), 512, Condicion(snr_db=10.0), rng)["iq"]
    assert detectar_ciego(iq)["detectada"]
    assert abs(snr_ciega_db(iq) - 10.0) < 3.0
