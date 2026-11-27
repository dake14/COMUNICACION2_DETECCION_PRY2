"""Genera señales nuevas para comparar arreglos. Uso: python -m experiments.gen_arreglos gnuradio|independiente"""
import importlib
import sys
from pathlib import Path

GENERADORES = {"gnuradio": ("experiments.gen_gnuradio_cascada", 2026111951),
               "independiente": ("experiments.gen_independiente", 2026111952)}

if __name__ == "__main__":
    nombre = sys.argv[1]
    sys.argv = sys.argv[:1]
    modulo, semilla = GENERADORES[nombre]
    gen = importlib.import_module(modulo)
    gen.DESTINO = Path("data/cascada_arreglos") / nombre
    gen.SEMILLA = semilla
    gen.POR_CASO = 16
    gen.main()
