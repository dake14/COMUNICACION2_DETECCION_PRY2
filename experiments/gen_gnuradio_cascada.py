"""Genera la prueba con GNU Radio. Ejecutar con el Python de radioconda."""
import json
import sys
from pathlib import Path

import numpy as np
from gnuradio import blocks, channels, digital, filter as gr_filter, gr
from gnuradio.filter import firdes

from src.cascada import geometria as g

DESTINO = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cascada_prueba/gnuradio")
SEMILLA = 2026111901
SNRS = (5, 10, 15, 20, 30)
POR_CASO = 8
FILTROS = 32


def correr(puntos, datos, sps, beta, taps, ruido, cfo, epsilon, semilla, n):
    tb = gr.top_block()
    fuente = blocks.vector_source_b(datos.tolist(), False)
    mapa = digital.chunks_to_symbols_bc([complex(p) for p in puntos], 1)
    rrc = firdes.root_raised_cosine(FILTROS, FILTROS, 1.0 / sps, beta, int(11 * sps * FILTROS) | 1)
    interp = gr_filter.pfb_arb_resampler_ccf(sps, rrc, FILTROS)
    canal = channels.channel_model(noise_voltage=ruido, frequency_offset=cfo, epsilon=epsilon,
                                   taps=[complex(t) for t in taps], noise_seed=semilla, block_tags=False)
    cabeza = blocks.head(gr.sizeof_gr_complex, n)
    sumidero = blocks.vector_sink_c()
    tb.connect(fuente, mapa, interp, canal, cabeza, sumidero)
    tb.run()
    return np.array(sumidero.data(), dtype=np.complex64)


def canales(rng):
    datos = json.loads(Path("configs/dataset_e2.json").read_text(encoding="utf-8"))["shift_channels"]
    lista = [[1.0 + 0j], *[[complex(a, b) for a, b in c] for c in datos]]
    return lista[int(rng.integers(len(lista)))]


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    grupos = {**{k: (v, k) for k, v in g.catalogo().items()},
              **{k: (v, "DESCONOCIDA") for k, v in g.desconocidas_prueba().items()}}
    manifiesto = []
    for nombre, (puntos, etiqueta) in grupos.items():
        for snr in SNRS:
            for _ in range(POR_CASO):
                sps = float(rng.choice([4.0, 5.5, 6.0, 8.0, 10.0]))
                n_sim = int(rng.choice([512, 1024]))
                n = int(n_sim * sps)
                beta = float(rng.choice([0.2, 0.35, 0.5]))
                taps = canales(rng)
                datos = rng.integers(0, len(puntos), n_sim + 80, dtype=np.uint8)
                cfo = float(rng.uniform(-0.01, 0.01))
                eps = 1.0 + float(rng.uniform(-1e-4, 1e-4))
                semilla = int(rng.integers(1, 2 ** 30))
                limpia = correr(puntos, datos, sps, beta, taps, 0.0, cfo, eps, semilla, n + 200)[200:]
                potencia = float(np.mean(np.abs(limpia) ** 2))
                ruido = float(np.sqrt(potencia / 10 ** (snr / 10)))
                iq = correr(puntos, datos, sps, beta, taps, ruido, cfo, eps, semilla, n + 200)[200:]
                archivo = f"g{len(manifiesto):04d}.c64"
                iq.tofile(DESTINO / archivo)
                manifiesto.append({"archivo": archivo, "real": nombre, "etiqueta": etiqueta, "snr": snr,
                                   "sps": sps, "beta": beta, "n_simbolos": n_sim})
    (DESTINO / "manifiesto.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    print(len(manifiesto), "archivos")


if __name__ == "__main__":
    main()
