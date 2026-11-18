"""Entrena el recuperador en CPU con datos procedurales renovados."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from src.cascada.geometria import geometria_aleatoria
from src.cascada.recuperador import Recuperador, a_tensor, guardar_recuperador, perdida_fase_invariante
from src.cascada.sintesis import condicion_aleatoria, transmitir

N_SIMBOLOS = 128


def lote(semilla, n):
    rng = np.random.default_rng(semilla)
    entrada = np.empty((n, N_SIMBOLOS * 8), np.complex64)
    objetivo = np.empty((n, N_SIMBOLOS * 8), np.complex64)
    for i in range(n):
        r = transmitir(geometria_aleatoria(rng), N_SIMBOLOS + 4, condicion_aleatoria(rng), rng)
        entrada[i], objetivo[i] = r["iq"][: N_SIMBOLOS * 8], r["limpia"][: N_SIMBOLOS * 8]
    return entrada, objetivo


def generar(semillas, por_lote, trabajos):
    partes = Parallel(n_jobs=trabajos)(delayed(lote)(int(s), por_lote) for s in semillas)
    return a_tensor(np.concatenate([p[0] for p in partes])), a_tensor(np.concatenate([p[1] for p in partes]))


@torch.inference_mode()
def validar(modelo, x, t):
    modelo.eval()
    return float(np.mean([perdida_fase_invariante(modelo(x[i:i + 64]), t[i:i + 64]).item()
                          for i in range(0, len(x), 64)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutos", type=float, default=55)
    p.add_argument("--pool", type=int, default=24000)
    p.add_argument("--salida", default="results/cascada/recuperador.pt")
    a = p.parse_args()
    torch.manual_seed(2026111801)
    torch.set_num_threads(8)
    secuencia = np.random.SeedSequence(2026111801)
    val_x, val_t = generar(secuencia.generate_state(12), 200, 8)
    modelo = Recuperador()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-5)
    inicio = time.time()
    limite = a.minutos * 60
    mejor, paso, ronda, historia = math.inf, 0, 0, []
    while time.time() - inicio < limite:
        ronda += 1
        semillas = np.random.SeedSequence([2026111802, ronda]).generate_state(24)
        x, t = generar(semillas, a.pool // 24, 8)
        modelo.train()
        orden = torch.randperm(len(x))
        acumulado, cuenta = 0.0, 0
        for i in range(0, len(x) - 31, 32):
            transcurrido = (time.time() - inicio) / limite
            if transcurrido >= 1:
                break
            for g in optimizador.param_groups:
                g["lr"] = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * transcurrido))
            k = orden[i:i + 32]
            perdida = perdida_fase_invariante(modelo(x[k]), t[k])
            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 2.0)
            optimizador.step()
            acumulado += perdida.item()
            cuenta += 1
            paso += 1
            if paso % 400 == 0:
                val = validar(modelo, val_x, val_t)
                modelo.train()
                fila = {"paso": paso, "min": round((time.time() - inicio) / 60, 1),
                        "train": round(acumulado / cuenta, 4), "val": round(val, 4)}
                historia.append(fila)
                print(json.dumps(fila), flush=True)
                acumulado, cuenta = 0.0, 0
                if val < mejor:
                    mejor = val
                    guardar_recuperador(modelo, a.salida, {"val": val, "paso": paso})
    Path(a.salida).with_suffix(".json").write_text(json.dumps({"mejor_val": mejor, "historia": historia}, indent=1),
                                                  encoding="utf-8")
    print("FIN", mejor, flush=True)


if __name__ == "__main__":
    main()
