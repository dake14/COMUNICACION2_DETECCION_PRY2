"""Afina el recuperador en GPU con canales variables, interferencias y el banco ampliado de canales."""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from src.cascada.geometria import geometria_aleatoria
from src.cascada.recuperador import a_tensor, cargar_recuperador, guardar_recuperador, perdida_fase_invariante
from src.cascada.sintesis_variada import transmitir_variado

N_SIMBOLOS = 128
LARGO = N_SIMBOLOS * 8
SEMILLA = 2026112401
BASE = Path("results/cascada/modelos/recuperador.pt")


# ──────────────────────────────  Datos  ──────────────────────────────
def lote(semilla, n):
    rng = np.random.default_rng(semilla)
    entrada = np.empty((n, LARGO), np.complex64)
    objetivo = np.empty((n, LARGO), np.complex64)
    canales = []
    for i in range(n):
        r = transmitir_variado(geometria_aleatoria(rng), N_SIMBOLOS + 4, rng)
        entrada[i], objetivo[i] = r["iq"][:LARGO], r["limpia"][:LARGO]
        canales.append(r["canal"])
    return entrada, objetivo, canales


def generar(semillas, por_lote, trabajos):
    partes = Parallel(n_jobs=trabajos)(delayed(lote)(int(s), por_lote) for s in semillas)
    x = a_tensor(np.concatenate([p[0] for p in partes]))
    t = a_tensor(np.concatenate([p[1] for p in partes]))
    return x, t, [c for p in partes for c in p[2]]


# ─────────────────────────────  Medición  ────────────────────────────
@torch.inference_mode()
def perdidas(modelo, x, t, disp):
    modelo.eval()
    salida = []
    for i in range(0, len(x), 128):
        y = modelo(x[i:i + 128].to(disp))
        objetivo = t[i:i + 128].to(disp)
        salida += [perdida_fase_invariante(y[k:k + 1], objetivo[k:k + 1]).item() for k in range(len(y))]
    return np.array(salida)


def por_canal(valores, canales):
    grupos = defaultdict(list)
    for v, c in zip(valores, canales):
        grupos[c].append(v)
    return {c: round(float(np.mean(v)), 4) for c, v in sorted(grupos.items())}


# ─────────────────────────────  Entrena  ─────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutos", type=float, default=70)
    p.add_argument("--pool", type=int, default=24000)
    p.add_argument("--trabajos", type=int, default=10)
    p.add_argument("--salida", default="results/cascada/modelos_robustos/recuperador.pt")
    a = p.parse_args()
    torch.manual_seed(SEMILLA)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semillas = np.random.SeedSequence(SEMILLA)
    val_x, val_t, val_c = generar(semillas.generate_state(16), 150, a.trabajos)
    modelo = cargar_recuperador(BASE).to(disp)
    antes = perdidas(modelo, val_x, val_t, disp)
    inicial = {"val": round(float(antes.mean()), 4), "por_canal": por_canal(antes, val_c)}
    print(json.dumps({"inicial": inicial}), flush=True)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-4, weight_decay=1e-5)
    inicio, limite = time.time(), a.minutos * 60
    mejor, paso, ronda, historia = float(antes.mean()), 0, 0, []
    while time.time() - inicio < limite:
        ronda += 1
        x, t, _ = generar(np.random.SeedSequence([SEMILLA, ronda]).generate_state(24), a.pool // 24, a.trabajos)
        modelo.train()
        orden = torch.randperm(len(x))
        acumulado, cuenta = 0.0, 0
        for _ in range(2):
            for i in range(0, len(x) - 63, 64):
                avance = (time.time() - inicio) / limite
                if avance >= 1:
                    break
                for g in optimizador.param_groups:
                    g["lr"] = 2e-5 + 0.5 * (3e-4 - 2e-5) * (1 + math.cos(math.pi * avance))
                k = orden[i:i + 64]
                perdida = perdida_fase_invariante(modelo(x[k].to(disp)), t[k].to(disp))
                optimizador.zero_grad(set_to_none=True)
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 2.0)
                optimizador.step()
                acumulado += perdida.item()
                cuenta += 1
                paso += 1
            orden = torch.randperm(len(x))
        val = float(perdidas(modelo, val_x, val_t, disp).mean())
        fila = {"ronda": ronda, "paso": paso, "min": round((time.time() - inicio) / 60, 1),
                "train": round(acumulado / max(cuenta, 1), 4), "val": round(val, 4)}
        historia.append(fila)
        print(json.dumps(fila), flush=True)
        if val < mejor:
            mejor = val
            guardar_recuperador(modelo.cpu(), a.salida, {"val": val, "paso": paso})
            modelo.to(disp)
    final = cargar_recuperador(a.salida).to(disp) if Path(a.salida).exists() else modelo
    despues = perdidas(final, val_x, val_t, disp)
    resumen = {"inicial": inicial, "final": {"val": round(float(despues.mean()), 4),
                                             "por_canal": por_canal(despues, val_c)}, "historia": historia}
    Path(a.salida).with_suffix(".json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    print("FIN", json.dumps(resumen["final"]), flush=True)


if __name__ == "__main__":
    main()
