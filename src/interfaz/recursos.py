# ─────────────────────────────  Imports  ─────────────────────────────
import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.cascada.pipeline import cargar_paquete

# ─────────────────────────────  Config  ──────────────────────────────
RAIZ = Path(__file__).resolve().parents[2]
MANIFIESTO_BANCO = RAIZ / "data" / "banco" / "base" / "manifiesto.jsonl"
MOTOR_DEFECTO = "red"


# ──────────────────────────────  Tipos  ──────────────────────────────
@dataclass(frozen=True)
class Motor:
    nombre: str
    descripcion: str
    ruta: Path


MOTORES = {
    "red": Motor("Red + refino", "El más eficiente de la comparación", RAIZ / "results/cascada/modelos/paquete_red.joblib"),
    "mixta": Motor("Mixta + refino", "Tres rutas de recuperación", RAIZ / "results/cascada/modelos/paquete.joblib"),
    "multi": Motor("Multigenerador", "Entrenado con todos los generadores y canales",
                   RAIZ / "results/cascada/modelos_multi/paquete.joblib"),
    "robusto": Motor("Canales variados", "32 canales, recuperador nuevo y CNN adversaria de dominio",
                     RAIZ / "results/cascada/modelos_robustos/paquete_adversaria_desplazada.joblib"),
}


# ─────────────────────────────  Modelos  ─────────────────────────────
class Modelos:
    def __init__(self):
        self._cargados = {}
        self._cerrojo = threading.Lock()

    def obtener(self, clave):
        if clave not in MOTORES:
            raise ValueError(f"Modelo desconocido: {clave}")
        if not MOTORES[clave].ruta.exists():
            raise ValueError(f"El modelo «{MOTORES[clave].nombre}» todavía no está disponible.")
        with self._cerrojo:
            if clave not in self._cargados:
                self._cargados[clave] = cargar_paquete(MOTORES[clave].ruta)
            return self._cargados[clave]

    @staticmethod
    def listado():
        return [{"id": k, "nombre": m.nombre, "descripcion": m.descripcion, "disponible": m.ruta.exists()}
                for k, m in MOTORES.items()]


# ──────────────────────────────  Banco  ──────────────────────────────
class Banco:
    def __init__(self, manifiesto=MANIFIESTO_BANCO):
        self.manifiesto = Path(manifiesto)
        self._filas = None

    @property
    def filas(self):
        if self._filas is None:
            texto = self.manifiesto.read_text(encoding="utf-8") if self.manifiesto.exists() else ""
            self._filas = {self._clave(f): f for f in map(json.loads, texto.splitlines())}
        return self._filas

    @staticmethod
    def _clave(f):
        return f["generador"], f["canal"], f["modulacion"], int(f["indice"])

    def opciones(self):
        if not self.filas:
            return None
        claves = list(self.filas)
        return {"generadores": sorted({c[0] for c in claves}), "canales": sorted({c[1] for c in claves}),
                "modulaciones": sorted({c[2] for c in claves}), "indices": sorted({c[3] for c in claves})}

    def ejemplo(self, generador=None, canal=None, modulacion=None, indice=None):
        if not self.filas:
            raise ValueError("El banco de señales no está generado.")
        if None in (generador, canal, modulacion, indice):
            fila = random.choice(list(self.filas.values()))
        else:
            fila = self.filas.get((generador, canal, modulacion, int(indice)))
            if fila is None:
                raise ValueError("Esa combinación no existe en el banco.")
        iq = np.fromfile(self.manifiesto.parent / fila["archivo"], np.complex64)
        return iq, {k: fila[k] for k in ("generador", "canal", "modulacion", "etiqueta", "indice", "sps", "beta",
                                         "pulso", "snr_db", "n_simbolos")}
