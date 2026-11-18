"""Genera vistas por la cadena elegida y entrena el descriptor contrastivo."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from experiments.comun_cascada import DIR_MODELOS, RUTA_DESCRIPTOR, RUTA_RECUPERADOR
from src.cascada import geometria as g
from src.cascada.cadena import recuperar
from src.cascada.descriptor import (Descriptor, aumentar, guardar_descriptor, imagen_densidad,
                                    perdida_contrastiva)
from src.cascada.recuperador import cargar_recuperador
from src.cascada.sintesis import condicion_aleatoria, transmitir

DATOS = Path("data/cascada")
VISTAS = 4
_MODELO = None


def _modelo(cadena):
    global _MODELO
    if cadena != "red":
        return None
    if _MODELO is None:
        torch.set_num_threads(1)
        _MODELO = cargar_recuperador(RUTA_RECUPERADOR)
    return _MODELO


def vista(puntos, rng, cadena):
    n = int(rng.choice([128, 256, 512, 1024, 2048], p=[0.1, 0.2, 0.3, 0.25, 0.15]))
    sps = float(rng.choice([8.0, 4.0, 5.3, 6.0, 10.0, 12.0], p=[0.6, 0.1, 0.075, 0.075, 0.075, 0.075]))
    cond = condicion_aleatoria(rng, snr_rango=(0.0, 30.0), cfo_max=0.008, prob_canal=0.6)
    iq = transmitir(puntos, n, cond, rng, sps_salida=sps)["iq"]
    r = recuperar(iq, cadena, _modelo(cadena), sps=sps)
    if r is None:
        return None
    return (np.clip(imagen_densidad(r["simbolos"]) / 1.5, 0, 1) * 255).astype(np.uint8)


def grupo(semilla, n_geometrias, cadena, puntos_fijos=None):
    rng = np.random.default_rng(semilla)
    imagenes, geo = [], []
    for _ in range(n_geometrias):
        puntos = puntos_fijos if puntos_fijos is not None else g.geometria_aleatoria(rng)
        vistas = [vista(puntos, rng, cadena) for _ in range(VISTAS)]
        if any(v is None for v in vistas):
            continue
        imagenes.append(np.stack(vistas))
        geo.append((math.log2(len(puntos)) / 7.0, min(g.contar_anillos(puntos), 6) / 6.0))
    return np.stack(imagenes), np.array(geo, np.float32)


def generar(n_total, cadena, semilla_base, trabajos=12):
    por_trabajo = 40
    semillas = np.random.SeedSequence(semilla_base).generate_state(n_total // por_trabajo).tolist()
    partes = Parallel(n_jobs=trabajos)(delayed(grupo)(int(s), por_trabajo, cadena) for s in semillas)
    return np.concatenate([p[0] for p in partes]), np.concatenate([p[1] for p in partes])


def lote_pares(x, geo, indices):
    a, b = torch.randint(0, VISTAS, (len(indices),)), torch.randint(1, VISTAS, (len(indices),))
    b = (a + b) % VISTAS
    fila = torch.as_tensor(indices)
    v1 = x[fila, a].float() / 255 * 1.5
    v2 = x[fila, b].float() / 255 * 1.5
    return aumentar(v1), aumentar(v2), geo[fila]


def evaluar(modelo, x, geo):
    modelo.eval()
    with torch.inference_mode():
        total, n = 0.0, 0
        for i in range(0, len(x) - 127, 128):
            indices = np.arange(i, i + 128)
            v1, v2, _ = lote_pares(x, geo, indices)
            total += perdida_contrastiva(modelo(v1)[0], modelo(v2)[0]).item()
            n += 1
    modelo.train()
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cadena", required=True)
    p.add_argument("--geometrias", type=int, default=3600)
    p.add_argument("--minutos", type=float, default=35)
    a = p.parse_args()
    torch.manual_seed(2026111804)
    torch.set_num_threads(1)
    DATOS.mkdir(parents=True, exist_ok=True)
    inicio_gen = time.time()
    x, geo = generar(a.geometrias, a.cadena, 2026111804)
    xv, geov = generar(a.geometrias // 9, a.cadena, 2026111805)
    print(json.dumps({"vistas": x.shape, "min_generacion": round((time.time() - inicio_gen) / 60, 1)}), flush=True)
    x, geo, xv, geov = map(torch.as_tensor, (x, geo, xv, geov))
    torch.set_num_threads(10)
    modelo = Descriptor()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    inicio = time.time()
    limite = a.minutos * 60
    mejor, paso, historia = math.inf, 0, []
    while time.time() - inicio < limite:
        orden = np.random.default_rng(paso).permutation(len(x))
        for i in range(0, len(orden) - 127, 128):
            avance = (time.time() - inicio) / limite
            if avance >= 1:
                break
            for grupo_opt in optimizador.param_groups:
                grupo_opt["lr"] = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * avance))
            v1, v2, g_real = lote_pares(x, geo, orden[i:i + 128])
            z1, g1 = modelo(v1)
            z2, g2 = modelo(v2)
            perdida = perdida_contrastiva(z1, z2) + 2.0 * (torch.nn.functional.smooth_l1_loss(g1, g_real)
                                                          + torch.nn.functional.smooth_l1_loss(g2, g_real))
            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            optimizador.step()
            paso += 1
            if paso % 150 == 0:
                val = evaluar(modelo, xv, geov)
                fila = {"paso": paso, "min": round((time.time() - inicio) / 60, 1),
                        "train": round(perdida.item(), 3), "val": round(val, 3)}
                historia.append(fila)
                print(json.dumps(fila), flush=True)
                if val < mejor:
                    mejor = val
                    guardar_descriptor(modelo, RUTA_DESCRIPTOR, {"val": val, "paso": paso, "cadena": a.cadena})
    (DIR_MODELOS / "descriptor.json").write_text(json.dumps({"mejor_val": mejor, "historia": historia}, indent=1),
                                                 encoding="utf-8")
    print("FIN", mejor, flush=True)


if __name__ == "__main__":
    main()
