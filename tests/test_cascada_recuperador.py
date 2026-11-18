import numpy as np
import torch

from src.cascada.recuperador import (Recuperador, a_tensor, aplicar_recuperador, cargar_recuperador,
                                     guardar_recuperador, perdida_fase_invariante)


def _ruido(n, semilla=0):
    rng = np.random.default_rng(semilla)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)


def test_conserva_longitud_y_tamano_razonable():
    m = Recuperador()
    assert sum(p.numel() for p in m.parameters()) < 300_000
    for n in (256, 1024, 1500):
        assert m(a_tensor(_ruido(n))).shape == (1, 2, n)


def test_perdida_invariante_a_fase_global():
    t = a_tensor(_ruido(512))
    z = _ruido(512) * 0 + (t[0, 0].numpy() + 1j * t[0, 1].numpy())
    girada = a_tensor(z * np.exp(0.9j))
    assert perdida_fase_invariante(girada, t).item() < 1e-6
    assert perdida_fase_invariante(a_tensor(_ruido(512, 1)), t).item() > 0.5


def test_aplicar_y_recargar(tmp_path):
    m = Recuperador()
    z = _ruido(600)
    a = aplicar_recuperador(m, z)
    ruta = tmp_path / "r.pt"
    guardar_recuperador(m, ruta)
    b = aplicar_recuperador(cargar_recuperador(ruta), z)
    assert a.shape == (600,) and np.allclose(a, b)
