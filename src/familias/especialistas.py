"""Especialistas: una red 1-D pequeña por vista, todas con la misma arquitectura."""
# ─────────────────────────────  Imports  ─────────────────────────────
import numpy as np
import torch
from torch import nn

# ─────────────────────────────  Config  ──────────────────────────────
ANCHO = 32
OCULTA = 128


# ──────────────────────────────  Red  ────────────────────────────────
def _bloque(entrada, salida, nucleo, paso):
    return nn.Sequential(nn.Conv1d(entrada, salida, nucleo, paso, nucleo // 2), nn.GroupNorm(4, salida), nn.SiLU())


class Especialista(nn.Module):
    def __init__(self, canales, salidas, ancho=ANCHO):
        super().__init__()
        self.config = {"canales": canales, "salidas": salidas, "ancho": ancho}
        self.cuerpo = nn.Sequential(_bloque(canales, ancho, 5, 1), _bloque(ancho, 2 * ancho, 5, 2),
                                    _bloque(2 * ancho, 4 * ancho, 5, 2), _bloque(4 * ancho, 4 * ancho, 3, 2))
        self.cabeza = nn.Sequential(nn.Linear(8 * ancho, OCULTA), nn.SiLU(), nn.Dropout(0.2),
                                    nn.Linear(OCULTA, salidas))

    def forward(self, x):
        h = self.cuerpo(x)
        return self.cabeza(torch.cat([h.mean(-1), h.amax(-1)], 1))


@torch.no_grad()
def log_probabilidades(modelo, x, lote=2048):
    """Log-softmax de las salidas para un arreglo (N, canales, largo)."""
    modelo.eval()
    disp = next(modelo.parameters()).device
    partes = [torch.log_softmax(modelo(torch.as_tensor(x[i:i + lote], dtype=torch.float32, device=disp)), 1).cpu()
              for i in range(0, len(x), lote)]
    return torch.cat(partes).numpy() if partes else np.zeros((0, modelo.config["salidas"]), np.float32)


def entrenar(x, y, salidas, disp, pasos=4000, lote=256, semilla=0, validacion=0.1):
    """Entrena con AdamW y devuelve el modelo con la mejor exactitud de validación interna."""
    rng = np.random.default_rng(semilla)
    torch.manual_seed(semilla)
    orden = rng.permutation(len(x))
    corte = max(1, int(len(x) * validacion))
    val, ent = orden[:corte], orden[corte:]
    modelo = Especialista(x.shape[1], salidas).to(disp)
    opt = torch.optim.AdamW(modelo.parameters(), 2e-3, weight_decay=1e-3)
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=pasos)
    perdida = nn.CrossEntropyLoss(label_smoothing=0.05)
    xv = torch.as_tensor(x[val], dtype=torch.float32, device=disp)
    yv = torch.as_tensor(y[val], device=disp)
    mejor, estado = -1.0, None
    for paso in range(pasos):
        modelo.train()
        i = ent[rng.integers(0, len(ent), lote)]
        xb = torch.as_tensor(x[i], dtype=torch.float32, device=disp)
        yb = torch.as_tensor(y[i], device=disp)
        opt.zero_grad()
        perdida(modelo(xb), yb).backward()
        opt.step()
        plan.step()
        if (paso + 1) % 500 == 0 or paso == pasos - 1:
            modelo.eval()
            with torch.no_grad():
                exactitud = float(sum((modelo(xv[k:k + 2048]).argmax(1) == yv[k:k + 2048]).sum()
                                      for k in range(0, len(xv), 2048)) / len(xv))
            if exactitud > mejor:
                mejor, estado = exactitud, {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    modelo.load_state_dict(estado)
    return modelo.eval(), mejor
