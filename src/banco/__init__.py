"""Banco de señales: cruza generadores, canales y configuraciones."""
from src.banco.canales import CANALES
from src.banco.config import Config, muestrear_config
from src.banco.generadores import GENERADORES
from src.banco.senal import gnuradio_disponible, sintetizar

__all__ = ["CANALES", "GENERADORES", "Config", "gnuradio_disponible", "muestrear_config", "sintetizar"]
