"""Catálogo de modulaciones digitales básicas: 18 clases, equivalencias y desconocidas propias."""
# ─────────────────────────────  Imports  ─────────────────────────────
from dataclasses import dataclass, replace

import numpy as np

from src.banco.cpm import TRANSMISORES as TRANSMISORES_CPM, niveles
from src.banco.generadores import GENERADORES
from src.cascada import geometria as g

# ─────────────────────────────  Config  ──────────────────────────────
DESCONOCIDA = "DESCONOCIDA"
FAMILIAS = ("amplitud", "fase", "qam", "apsk", "frecuencia", "otra")
EQUIVALENTES = {
    "BPSK": ("DBPSK",),
    "QPSK": ("DQPSK",),
    "OOK": ("2ASK unipolar",),
    "MSK": ("OQPSK con pulso de medio seno", "2FSK de fase continua con h = 0,5"),
}


# ──────────────────────────────  Tipos  ──────────────────────────────
@dataclass(frozen=True)
class Modulacion:
    nombre: str
    familia: str
    tipo: str
    puntos: tuple = ()
    orden: int = 0
    h: tuple = ()
    bt: tuple = ()

    def transmitir(self, generador, cfg, n, rng):
        """Banda base a `cfg.sps` muestras por símbolo con `n` símbolos y los parámetros sorteados."""
        crear = GENERADORES[generador].crear
        if self.tipo == "lineal":
            puntos = np.asarray(self.puntos)
            return crear(puntos[rng.integers(0, len(puntos), n)], cfg, rng), {}
        if self.tipo == "pi4dqpsk":
            saltos = np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n)
            return crear(np.exp(1j * np.cumsum(saltos)), cfg, rng), {}
        if self.tipo == "oqpsk":
            return _oqpsk(np.asarray(self.puntos), crear, cfg, n, rng)
        if self.tipo == "cpm":
            h = float(rng.uniform(*self.h)) if self.h[0] != self.h[1] else float(self.h[0])
            bt = float(rng.choice(self.bt)) if self.bt else None
            x = TRANSMISORES_CPM[generador](niveles(self.orden, n, rng), h, bt, cfg)
            return x, {"h": h, "bt": bt}
        raise ValueError(f"Tipo de modulación desconocido: {self.tipo}")


# ─────────────────────────────  Formas  ──────────────────────────────
def retardar(x, d):
    """Retardo fraccionario de `d` muestras por FFT, con relleno para no envolver."""
    n = len(x)
    largo = 2 * n
    f = np.fft.fftfreq(largo)
    return np.fft.ifft(np.fft.fft(x, largo) * np.exp(-2j * np.pi * f * d))[:n]


def _oqpsk(puntos, crear, cfg, n, rng):
    if cfg.pulso == "semiseno":
        cfg = replace(cfg, pulso="rrc")
    s = puntos[rng.integers(0, len(puntos), n)]
    i = np.asarray(crear(s.real.astype(np.complex128), cfg, rng))
    q = np.asarray(crear(s.imag.astype(np.complex128), cfg, rng))
    largo = min(len(i), len(q))
    return i[:largo] + 1j * retardar(q[:largo], cfg.sps / 2).real, {"pulso": cfg.pulso}


def unipolar(valores):
    z = np.asarray(valores, dtype=np.complex128)
    return z / np.sqrt(np.mean(np.abs(z) ** 2))


def _lineal(nombre, familia, puntos):
    return Modulacion(nombre, familia, "lineal", tuple(np.asarray(puntos, dtype=np.complex128)))


def _fsk(nombre, orden, h, bt=()):
    return Modulacion(nombre, "frecuencia", "cpm", orden=orden, h=h, bt=bt)


# ─────────────────────────────  Catálogo  ────────────────────────────
def catalogo_digital():
    prueba = g.desconocidas_prueba()
    clases = [
        _lineal("OOK", "amplitud", unipolar([0, 1])),
        _lineal("4ASK", "amplitud", unipolar([0, 1, 2, 3])),
        _lineal("4PAM", "amplitud", g.pam(4)),
        _lineal("BPSK", "fase", g.psk(2)),
        _lineal("QPSK", "fase", g.psk(4, np.pi / 4)),
        _lineal("8PSK", "fase", g.psk(8)),
        Modulacion("OQPSK", "fase", "oqpsk", tuple(g.psk(4, np.pi / 4))),
        Modulacion("PI4DQPSK", "fase", "pi4dqpsk"),
        _lineal("16QAM", "qam", g.qam(16)),
        _lineal("32QAM", "qam", prueba["32QAM"]),
        _lineal("64QAM", "qam", g.qam(64)),
        _lineal("256QAM", "qam", g.qam(256)),
        _lineal("16APSK", "apsk", prueba["16APSK"]),
        _lineal("32APSK", "apsk", prueba["32APSK"]),
        _fsk("2FSK", 2, (0.7, 1.5)),
        _fsk("4FSK", 4, (0.25, 1.0)),
        _fsk("MSK", 2, (0.5, 0.5)),
        _fsk("GMSK", 2, (0.5, 0.5), (0.25, 0.3, 0.5)),
    ]
    return {m.nombre: m for m in clases}


def desconocidas_validacion_digital():
    familias = {"6PSK": "fase", "24QAM": "qam", "HEX19": "otra", "ANILLOS12": "apsk"}
    salida = {k: _lineal(k, familias[k], v) for k, v in g.desconocidas_validacion().items()}
    salida["3FSK"] = _fsk("3FSK", 3, (0.5, 1.0))
    return salida


def desconocidas_prueba_digital():
    familias = {"128QAM": "qam", "12PSK": "fase", "8QAM_ESTRELLA": "apsk", "ARB_A": "otra", "ARB_B": "otra"}
    salida = {k: _lineal(k, f, g.desconocidas_prueba()[k]) for k, f in familias.items()}
    salida["8FSK"] = _fsk("8FSK", 8, (0.25, 0.6))
    return salida


def modulaciones_digitales():
    """{nombre: (Modulacion, etiqueta, grupo)} como `armar_banco.modulaciones()`."""
    salida = {k: (v, k, "catalogo") for k, v in catalogo_digital().items()}
    salida.update({k: (v, DESCONOCIDA, "validacion") for k, v in desconocidas_validacion_digital().items()})
    salida.update({k: (v, DESCONOCIDA, "prueba") for k, v in desconocidas_prueba_digital().items()})
    return salida
