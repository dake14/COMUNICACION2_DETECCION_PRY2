"""Configuración de transmisión de cada archivo del banco."""
from dataclasses import asdict, dataclass

from src.banco.generadores import GENERADORES

SPS_ENTEROS = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
SPS_FRACCIONARIOS = (3.5, 5.5, 6.4, 7.3, 9.6)
SPS_EXTREMOS_ENTEROS = (16.0, 20.0, 24.0, 32.0)
SPS_EXTREMOS_FRACCIONARIOS = (2.5, 2.8)
BETAS = (0.15, 0.2, 0.25, 0.35, 0.5, 0.7)
SPANS = (6, 8, 10, 12, 16)
N_SIMBOLOS = (256, 512, 1024, 2048)
SNRS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
CFO_MAX = 0.012
CFO_POR_SPS = 0.08


@dataclass(frozen=True)
class Config:
    n_simbolos: int
    sps: float
    pulso: str
    beta: float
    span: int
    snr_db: float
    cfo: float

    def como_dict(self):
        return asdict(self)


def muestrear_config(generador, rng, n_simbolos=N_SIMBOLOS, snrs=SNRS, extremos=0.0):
    info = GENERADORES[generador]
    opciones = SPS_ENTEROS + SPS_FRACCIONARIOS if info.fraccionario else SPS_ENTEROS
    if extremos > 0 and rng.uniform() < extremos:
        opciones = SPS_EXTREMOS_ENTEROS + (SPS_EXTREMOS_FRACCIONARIOS if info.fraccionario else ())
    sps = float(rng.choice(opciones))
    limite = min(CFO_MAX, CFO_POR_SPS / sps)
    return Config(n_simbolos=int(rng.choice(n_simbolos)), sps=sps, pulso=str(rng.choice(info.pulsos)),
                  beta=float(rng.choice(BETAS)), span=int(rng.choice(SPANS)), snr_db=float(rng.choice(snrs)),
                  cfo=float(rng.uniform(-limite, limite)))
