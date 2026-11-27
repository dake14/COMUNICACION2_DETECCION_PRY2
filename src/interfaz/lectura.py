# ─────────────────────────────  Imports  ─────────────────────────────
import io
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.monitor import load_signal

# ─────────────────────────────  Config  ──────────────────────────────
FORMATOS = {
    "auto": "Detectar por extensión",
    "complex64": "Complejo de 32 bits (GNU Radio)",
    "complex128": "Complejo de 64 bits",
    "int16": "Enteros I/Q de 16 bits",
    "int8": "Enteros I/Q de 8 bits",
    "uint8": "Enteros I/Q sin signo (RTL-SDR)",
}
POR_EXTENSION = {
    ".c64": "complex64", ".cfile": "complex64", ".fc32": "complex64", ".cf32": "complex64", ".bin": "complex64",
    ".dat": "complex64", ".iq": "complex64", ".raw": "complex64", ".cs16": "int16", ".sc16": "int16",
    ".cs8": "int8", ".sc8": "int8", ".cu8": "uint8",
}
CON_ESTRUCTURA = (".npy", ".npz", ".csv")


# ───────────────────────────────  Core  ──────────────────────────────
def _pares(datos, tipo, centro, escala):
    v = np.frombuffer(datos, dtype=tipo)
    if len(v) % 2:
        raise ValueError("El archivo tiene un número impar de valores y no forma pares I/Q.")
    v = (v.astype(np.float32) - centro) / escala
    return v[::2] + 1j * v[1::2]


def _binario(datos, formato):
    if formato in ("complex64", "complex128"):
        tamano = np.dtype(formato).itemsize
        if len(datos) % tamano:
            raise ValueError(f"El tamaño del archivo no es múltiplo de {tamano} bytes; revisa el formato elegido.")
        return np.frombuffer(datos, dtype=formato).astype(np.complex64)
    if formato == "int16":
        return _pares(datos, np.int16, 0.0, 32768.0)
    if formato == "int8":
        return _pares(datos, np.int8, 0.0, 128.0)
    if formato == "uint8":
        return _pares(datos, np.uint8, 127.5, 127.5)
    raise ValueError(f"Formato desconocido: {formato}")


def _wav(datos):
    _, audio = wavfile.read(io.BytesIO(datos))
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("El WAV debe tener dos canales: I y Q.")
    audio = audio.astype(np.float32)
    return (audio[:, 0] + 1j * audio[:, 1]) / max(float(np.abs(audio).max()), 1e-12)


def _con_estructura(datos, extension):
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = Path(carpeta) / f"senal{extension}"
        ruta.write_bytes(datos)
        return np.asarray(load_signal(ruta), dtype=np.complex64)


def leer_senal(datos, nombre, formato="auto"):
    extension = Path(nombre or "").suffix.lower()
    if not datos:
        raise ValueError("El archivo está vacío.")
    if formato not in FORMATOS:
        raise ValueError(f"Formato desconocido: {formato}")
    if formato != "auto":
        return _binario(datos, formato)
    if extension == ".wav":
        return _wav(datos)
    if extension in CON_ESTRUCTURA:
        return _con_estructura(datos, extension)
    if extension in POR_EXTENSION:
        return _binario(datos, POR_EXTENSION[extension])
    raise ValueError(f"No reconozco la extensión «{extension}». Elige el formato manualmente.")
