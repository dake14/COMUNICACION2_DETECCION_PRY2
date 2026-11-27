# ─────────────────────────────  Imports  ─────────────────────────────
import http.client
import io
import json
import threading

import numpy as np
import pytest
from scipy.io import wavfile

from src.banco import Config, sintetizar
from src.cascada import geometria as g
from src.cascada.cadena import recuperar
from src.interfaz.analisis import analizar_senal
from src.interfaz.graficas import alinear, trazas_ojo
from src.interfaz.lectura import leer_senal
from src.interfaz.recuperacion import recuperar_detalle
from src.interfaz.recursos import MOTORES, Banco, Modelos
from src.interfaz.servidor import crear_servidor

# ─────────────────────────────  Config  ──────────────────────────────
HAY_MODELO = MOTORES["red"].ruta.exists()
HAY_INTEGRADO = MOTORES["integrado"].ruta.exists()


# ─────────────────────────────  Ayudas  ──────────────────────────────
def senal(modulacion="QPSK", snr=35.0, sps=8.0, n=1024):
    cfg = Config(n, sps, "rrc", 0.35, 10, snr, 0.001)
    return sintetizar({**g.desconocidas_prueba(), **g.catalogo()}[modulacion], "procedural", "ideal", cfg, np.random.default_rng(3))["iq"]


@pytest.fixture
def servidor():
    s = crear_servidor(0, Modelos(), Banco("no_existe.jsonl"))
    hilo = threading.Thread(target=s.serve_forever, daemon=True)
    hilo.start()
    yield s
    s.shutdown()
    s.server_close()


def pedir(servidor, metodo, ruta, cuerpo=None, cabeceras=None):
    c = http.client.HTTPConnection("127.0.0.1", servidor.server_address[1], timeout=60)
    c.request(metodo, ruta, cuerpo, cabeceras or {})
    r = c.getresponse()
    datos = r.read()
    c.close()
    return r.status, datos


# ─────────────────────────────  Lectura  ─────────────────────────────
def test_lee_complejos_por_extension():
    x = np.array([1 + 2j, 3 - 4j], np.complex64)
    assert np.array_equal(leer_senal(x.tobytes(), "a.c64"), x)


def test_lee_enteros_iq():
    assert leer_senal(np.array([16384, -16384], np.int16).tobytes(), "a.cs16")[0] == pytest.approx(0.5 - 0.5j)
    assert leer_senal(bytes([255, 0]), "a.cu8")[0] == pytest.approx(1 - 1j)


def test_lee_npy_y_wav():
    x = np.arange(4, dtype=np.complex64) + 1j
    b = io.BytesIO()
    np.save(b, x)
    assert np.array_equal(leer_senal(b.getvalue(), "a.npy"), x)
    b = io.BytesIO()
    wavfile.write(b, 8000, np.array([[1000, -1000], [2000, 0]], np.int16))
    assert leer_senal(b.getvalue(), "a.wav")[1] == pytest.approx(1 + 0j)


@pytest.mark.parametrize("datos,nombre,formato", [(b"", "a.c64", "auto"), (b"123", "a.c64", "auto"),
                                                  (b"12", "a.xyz", "auto"), (b"123", "a", "int16")])
def test_rechaza_datos_invalidos(datos, nombre, formato):
    with pytest.raises(ValueError):
        leer_senal(datos, nombre, formato)


# ─────────────────────────────  Gráficas  ────────────────────────────
def test_detalle_coincide_con_la_cadena_principal():
    iq = senal()
    a = recuperar_detalle(iq, "directa")
    b = recuperar(iq, "directa")
    assert np.allclose(a["simbolos"], b["simbolos"]) and a["df"] == pytest.approx(b["df"])


def test_ojo_abierto_en_el_instante_de_decision():
    r = recuperar_detalle(senal(), "directa")
    ojo = trazas_ojo(r)
    centro = len(ojo["t"]) // 2
    assert ojo["t"][centro] == 0
    for eje in ("i", "q"):
        v = np.abs(np.array(ojo[eje])[:, centro])
        assert v.mean() > 0.45 and v.std() < 0.2 * v.mean()


@pytest.mark.parametrize("ruta", ["directa", "clasica"])
def test_centro_del_ojo_son_los_simbolos_de_la_constelacion(ruta):
    r = recuperar_detalle(senal("16QAM", snr=18.0), ruta)
    ojo = trazas_ojo(r)
    centro = len(ojo["t"]) // 2
    z = np.array(ojo["i"])[:, centro] + 1j * np.array(ojo["q"])[:, centro]
    distancia = np.abs(z[:, None] - alinear(r["simbolos"])[None, :]).min(axis=1)
    assert distancia.max() < 0.01


def test_constelacion_alineada_con_los_ejes():
    z = alinear(g.catalogo()["BPSK"][np.random.default_rng(1).integers(0, 2, 500)] * np.exp(0.7j))
    assert np.abs(z.imag).max() < 0.05


# ─────────────────────────────  Análisis  ────────────────────────────
@pytest.mark.skipif(not HAY_MODELO, reason="modelos locales no disponibles")
def test_analisis_entrega_lo_que_dibuja_la_interfaz():
    d = analizar_senal(senal(), Modelos().obtener("red"))
    assert sum(p["valor"] for p in d["probabilidades"]) == pytest.approx(1, abs=0.01)
    assert len(d["cnn"]["huella"]) == 128 and np.shape(d["cnn"]["entrada"]["polar"]) == (64, 64)
    assert len(d["constelacion"]["x"]) > 500 and np.shape(d["ojo"]["i"])[1] == len(d["ojo"]["t"])
    json.dumps(d, allow_nan=False)


@pytest.mark.skipif(not HAY_INTEGRADO, reason="paquete integrado no disponible")
def test_el_motor_integrado_entrega_familias_y_especialistas():
    d = analizar_senal(senal("16QAM"), Modelos().obtener("integrado"))
    assert sum(p["valor"] for p in d["probabilidades"]) == pytest.approx(1, abs=0.01)
    assert len(d["probabilidades"]) == 19 and {p["nombre"] for p in d["probabilidades"]} >= {"256QAM", "GMSK"}
    e = d["enrutador"]
    assert [f["nombre"] for f in e["familias"]][0] == "qam" and len(e["especialistas"]) == 5
    assert e["rama"] == "constelacion" and e["tonos"] is None and len(d["cnn"]["huella"]) == 128
    assert d["decision"]["mas_probable"] == "16QAM" and d["decision"]["misma_senal_que"] == []
    assert all(a["nombre"] != "16QAM" for a in d["decision"]["tambien_puede_ser"])
    json.dumps(d, allow_nan=False)


@pytest.mark.skipif(not HAY_INTEGRADO, reason="paquete integrado no disponible")
def test_el_motor_integrado_consulta_al_especialista_de_constelacion():
    paquete = Modelos().obtener("integrado")
    assert paquete["grupos"] == [("qam", "apsk")] and paquete["especialista_modelo"][1][-2:] == ["16APSK", "32APSK"]
    d = analizar_senal(senal("16APSK", n=4096), paquete)
    voto = [e for e in d["enrutador"]["especialistas"] if e["vista"] == "constelacion"]
    assert voto and voto[0]["clase"] == "16APSK" and d["decision"]["mas_probable"] == "16APSK"
    bpsk = analizar_senal(senal("BPSK"), paquete)["enrutador"]
    assert bpsk["familias"][0]["nombre"] == "fase" and all(e["vista"] != "constelacion" for e in bpsk["especialistas"])


@pytest.mark.skipif(not HAY_INTEGRADO, reason="paquete integrado no disponible")
def test_una_constelacion_que_no_cabe_sube_la_desconocida():
    paquete = Modelos().obtener("integrado")
    assert "consistencia" in paquete and "8PSK" in paquete["plantillas"]
    limpia = analizar_senal(senal("8PSK", n=4096), paquete)["decision"]
    assert limpia["modulacion"] == "8PSK" and limpia["no_cabe_en"] == []
    rara = analizar_senal(senal("12PSK", n=4096), paquete)["decision"]
    assert rara["no_cabe_en"] and rara["modulacion"] in ("DESCONOCIDA", "INDETERMINADA")


@pytest.mark.skipif(not HAY_INTEGRADO, reason="paquete integrado no disponible")
def test_el_motor_integrado_avisa_de_las_senales_identicas():
    d = analizar_senal(senal("QPSK"), Modelos().obtener("integrado"))
    assert d["decision"]["modulacion"] == "QPSK" and d["decision"]["misma_senal_que"] == ["DQPSK"]


def test_rechaza_senal_corta():
    with pytest.raises(ValueError):
        analizar_senal(np.ones(20, np.complex64), {})


# ─────────────────────────────  Servidor  ────────────────────────────
def test_sirve_pagina_y_estado(servidor):
    assert pedir(servidor, "GET", "/")[0] == 200
    assert pedir(servidor, "GET", "/app.js")[0] == 200
    estado, datos = pedir(servidor, "GET", "/api/estado")
    assert estado == 200 and json.loads(datos)["banco"] is None


@pytest.mark.parametrize("ruta", ["/../servidor.py", "/web/app.js", "/api/nada"])
def test_no_expone_otros_archivos(servidor, ruta):
    assert pedir(servidor, "GET", ruta)[0] == 404


def test_rechaza_otros_origenes(servidor):
    assert pedir(servidor, "POST", "/api/analizar", b"x", {"Origin": "https://ejemplo.com"})[0] == 403
    assert pedir(servidor, "GET", "/api/estado", None, {"Host": "ejemplo.com"})[0] == 403


def test_no_abre_dos_servidores_en_el_mismo_puerto(servidor):
    with pytest.raises(OSError):
        crear_servidor(servidor.server_address[1], Modelos(), Banco("no_existe.jsonl"))


def test_error_de_formato_es_claro(servidor):
    estado, datos = pedir(servidor, "POST", "/api/analizar?nombre=a.xyz", b"1234")
    assert estado == 400 and "extensión" in json.loads(datos)["error"]


@pytest.mark.skipif(not HAY_MODELO, reason="modelos locales no disponibles")
def test_analiza_un_archivo_subido(servidor):
    estado, datos = pedir(servidor, "POST", "/api/analizar?nombre=a.c64&motor=red", senal().tobytes())
    d = json.loads(datos)
    assert estado == 200 and d["nombre"] == "a.c64" and d["etapas"]["sps"] == pytest.approx(8.0)
