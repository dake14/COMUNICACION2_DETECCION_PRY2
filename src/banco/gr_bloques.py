"""Transmisores y canales hechos con bloques de GNU Radio."""
import math

import numpy as np
from gnuradio import analog, blocks, channels, filter as gr_filter, gr
from gnuradio.filter import firdes

from src.banco.canales import doppler_por_simbolo, perfil_rayos
from src.banco.cpm import pulso_frecuencia
from src.banco.isi import ajustar

FILTROS = 32
OSCILADORES = 8
SOBRE_CPM = 32


def _correr(x, *bloques):
    tb = gr.top_block()
    fuente = blocks.vector_source_c(np.asarray(x, dtype=np.complex64).tolist(), False)
    cabeza = blocks.head(gr.sizeof_gr_complex, 2 * len(x) * FILTROS)
    sumidero = blocks.vector_sink_c()
    tb.connect(fuente, *bloques, cabeza, sumidero)
    tb.run()
    return np.array(sumidero.data(), dtype=np.complex128)


def _taps(cfg, ganancia, muestras_por_simbolo):
    n = (cfg.span * int(muestras_por_simbolo)) | 1
    if cfg.pulso == "gauss":
        return firdes.gaussian(ganancia, muestras_por_simbolo, 0.3 + 0.5 * cfg.beta, n)
    return firdes.root_raised_cosine(ganancia, muestras_por_simbolo, 1.0, cfg.beta, n)


def tx_polifase(simbolos, cfg):
    return _correr(simbolos, gr_filter.pfb_arb_resampler_ccf(cfg.sps, _taps(cfg, FILTROS, FILTROS), FILTROS))


def tx_fir_mmse(simbolos, cfg):
    entero = max(2, math.ceil(cfg.sps))
    bloques = [gr_filter.interp_fir_filter_ccf(entero, _taps(cfg, entero, entero))]
    if entero != cfg.sps:
        bloques.append(gr_filter.mmse_resampler_cc(0.0, entero / cfg.sps))
    return _correr(simbolos, *bloques)


def _correr_fm(niveles, *bloques):
    tb = gr.top_block()
    fuente = blocks.vector_source_f(np.asarray(niveles, dtype=np.float32).tolist(), False)
    sumidero = blocks.vector_sink_c()
    tb.connect(fuente, *bloques, sumidero)
    tb.run()
    return np.array(sumidero.data(), dtype=np.complex128)


def _modulador_fm(h, bt, sobre):
    taps = [float(v) for v in pulso_frecuencia(sobre, bt)]
    return [gr_filter.interp_fir_filter_fff(sobre, taps), analog.frequency_modulator_fc(math.pi * h / sobre)]


def tx_cpm_polifase(niveles, h, bt, cfg):
    """Fase continua a 32 muestras por símbolo y remuestreo polifásico a la tasa pedida."""
    r = cfg.sps / SOBRE_CPM
    prototipo = firdes.low_pass(FILTROS, FILTROS, 0.4 * r, 0.1 * r)
    return _correr_fm(niveles, *_modulador_fm(h, bt, SOBRE_CPM),
                      gr_filter.pfb_arb_resampler_ccf(r, prototipo, FILTROS))


def tx_cpm_fir_mmse(niveles, h, bt, cfg):
    entero = max(2, math.ceil(cfg.sps))
    bloques = _modulador_fm(h, bt, entero)
    if entero != cfg.sps:
        bloques.append(gr_filter.mmse_resampler_cc(0.0, entero / cfg.sps))
    return _correr_fm(niveles, *bloques)


def _completar(y, n):
    return np.r_[y, np.zeros(n - len(y), dtype=np.complex128)][:n] if len(y) < n else y[:n]


def _perfil(sps, rng):
    d, m = perfil_rayos(sps, rng, 3.0, 3, 6)
    return d, ajustar(m), int(math.ceil(d[-1])) + 8, int(rng.integers(1, 2 ** 30))


def canal_multitrayecto(x, sps, rng):
    d, m, largo, semilla = _perfil(sps, rng)
    taps = np.zeros(largo, dtype=np.complex128)
    for retardo, magnitud in zip(np.round(d).astype(int), m):
        taps[retardo] += magnitud * np.exp(2j * np.pi * rng.uniform())
    epsilon = 1 + float(rng.uniform(-1e-4, 1e-4))
    bloque = channels.channel_model(0.0, 0.0, epsilon, [complex(t) for t in taps], semilla, False)
    return _completar(_correr(x, bloque), len(x)), {"retardos": d.tolist(), "epsilon": epsilon}


def canal_selectivo(x, sps, rng):
    d, m, largo, semilla = _perfil(sps, rng)
    fd = doppler_por_simbolo(rng)
    los = bool(rng.integers(2))
    k = float(rng.uniform(1, 10))
    bloque = channels.selective_fading_model(OSCILADORES, fd / sps, los, k, semilla, d.tolist(), m.tolist(), largo)
    return _completar(_correr(x, bloque), len(x)), {"retardos": d.tolist(), "doppler_por_simbolo": fd, "los": los, "k": k}


def canal_dinamico(x, sps, rng):
    d, m, largo, semilla = _perfil(sps, rng)
    fd = doppler_por_simbolo(rng)
    los = bool(rng.integers(2))
    k = float(rng.uniform(1, 10))
    sro = float(rng.uniform(1e-5, 1e-4))
    cfo = float(rng.uniform(1e-4, 1e-3))
    bloque = channels.dynamic_channel_model(1.0, sro / 100, sro, cfo / 100, cfo, OSCILADORES, fd / sps, los, k,
                                            d.tolist(), m.tolist(), largo, 0.0, semilla)
    return _completar(_correr(x, bloque), len(x)), {"doppler_por_simbolo": fd, "los": los, "k": k, "sro": sro, "cfo": cfo}


def canal_imperfecciones(x, sps, rng):
    p = {"phase_noise_mag": float(rng.uniform(-45, -25)), "magbal": float(rng.uniform(0.1, 1.0)),
         "phasebal": float(rng.uniform(0.5, 5.0)), "q_ofs": float(rng.uniform(-0.03, 0.03)),
         "i_ofs": float(rng.uniform(-0.03, 0.03)), "freq_offset": 0.0,
         "gamma": float(rng.uniform(0, 0.02)), "beta": float(rng.uniform(0, 0.02))}
    return _completar(_correr(x / np.sqrt(np.mean(np.abs(x) ** 2)), channels.impairments(**p)), len(x)), p
