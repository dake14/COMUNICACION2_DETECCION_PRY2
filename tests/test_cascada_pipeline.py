import json
import subprocess
import sys

import numpy as np
import pytest

from src.cascada import geometria as g
from src.cascada.decisor import DESCONOCIDA, INDETERMINADA, Decisor
from src.cascada.descriptor import Descriptor, calcular_huellas, guardar_descriptor, imagen_densidad
from src.cascada.lote import agrupar_desconocidas, fila_tabla
from src.cascada.pipeline import analizar_cascada, cargar_paquete, clasificacion_monitor, guardar_paquete
from src.cascada.sintesis import Condicion, transmitir


@pytest.fixture(scope="module")
def paquete(tmp_path_factory):
    base = tmp_path_factory.mktemp("paquete")
    rng = np.random.default_rng(0)
    modelo = Descriptor()
    guardar_descriptor(modelo, base / "d.pt")

    def huellas(puntos, n):
        imagenes = []
        for _ in range(n):
            s = puntos[rng.integers(0, len(puntos), 300)]
            s = s + 0.05 * (rng.normal(size=300) + 1j * rng.normal(size=300))
            imagenes.append(imagen_densidad(s * np.exp(1j * rng.uniform(-3, 3))))
        return calcular_huellas(modelo, np.stack(imagenes))[0]

    catalogo = {k: huellas(v, 30) for k, v in g.catalogo().items()}
    fondo = np.concatenate([huellas(g.geometria_aleatoria(rng), 3) for _ in range(60)])
    decisor = Decisor(dim_max=8, componentes=1, componentes_fondo=3).ajustar(fondo, catalogo)
    decisor.temperatura, decisor.umbral = 5.0, 0.9
    guardar_paquete(base / "p.joblib", decisor, "directa", None, "d.pt")
    return base / "p.joblib"


def _señal(nombre="QPSK", snr=20.0, sps=8):
    rng = np.random.default_rng(5)
    return transmitir(g.catalogo()[nombre], 512, Condicion(snr_db=snr), rng, sps_salida=sps)["iq"]


def test_salida_completa(paquete):
    r = analizar_cascada(_señal(), cargar_paquete(paquete))
    assert r["estado"] == "clasificada"
    assert np.isclose(sum(r["probabilidades"].values()), 1.0)
    assert r["etapas"]["sps_origen"] == 8.0
    assert r["deteccion"]["detectada"] is True


def test_senal_corta_no_falla(paquete):
    r = analizar_cascada(_señal()[:255], cargar_paquete(paquete))
    assert r["modulacion"] == INDETERMINADA and r["estado"] == "senal_muy_corta"


def test_senal_nula_no_falla(paquete):
    r = analizar_cascada(np.zeros(1024, complex), cargar_paquete(paquete))
    assert r["modulacion"] == INDETERMINADA


def test_escala_no_cambia_la_decision(paquete):
    p = cargar_paquete(paquete)
    iq = _señal("16QAM")
    base = analizar_cascada(iq, p)
    for k in (0.01, 100.0):
        r = analizar_cascada(iq * k, p)
        assert r["modulacion"] == base["modulacion"]
        assert r["confianza"] == pytest.approx(base["confianza"], abs=1e-3)


def test_ruido_y_tono_se_detectan_sin_fallar(paquete):
    p = cargar_paquete(paquete)
    rng = np.random.default_rng(9)
    ruido = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    assert analizar_cascada(ruido, p)["deteccion"]["detectada"] is False
    tono = np.exp(2j * np.pi * 0.05 * np.arange(4096)) + 0.01 * ruido
    assert analizar_cascada(tono, p)["deteccion"]["detectada"] is True


def test_contrato_monitor_y_tabla(paquete):
    p = cargar_paquete(paquete)
    assert clasificacion_monitor(_señal(), False, p)["modulation"] == "NO_SIGNAL"
    c = clasificacion_monitor(_señal(), None, p)
    assert c["motor"] == "cascada" and "probabilidades" in c
    r = analizar_cascada(_señal(), p)
    r["senal"] = "X01"
    r["grupo"] = None
    fila = fila_tabla(r)
    assert fila["Señal"] == "X01" and fila["Señal detectada"] == "Sí"


def test_agrupar_desconocidas(paquete):
    p = cargar_paquete(paquete)
    z = np.ones(128) / np.sqrt(128)
    otra = np.r_[np.ones(64), -np.ones(64)] / np.sqrt(128)
    r = [{"modulacion": DESCONOCIDA, "huella": h} for h in (z, z, otra, otra)]
    r = agrupar_desconocidas(r, p)
    assert r[0]["grupo"] == r[1]["grupo"] is not None
    assert r[2]["grupo"] == r[3]["grupo"] is not None
    assert r[0]["grupo"] != r[2]["grupo"]


def test_monitor_cli_con_cascada(paquete, tmp_path):
    entrada = tmp_path / "x.c64"
    _señal().astype(np.complex64).tofile(entrada)
    salida = tmp_path / "x.json"
    subprocess.run([sys.executable, "-m", "src.monitor", "--input", str(entrada), "--motor", "cascada",
                    "--paquete", str(paquete), "--output", str(salida),
                    "--plot", str(tmp_path / "x.png")], check=True, capture_output=True)
    datos = json.loads(salida.read_text(encoding="utf-8"))
    assert datos["classification"]["motor"] == "cascada"
    assert not any("Modelo clasificador" in w for w in datos["warnings"])
