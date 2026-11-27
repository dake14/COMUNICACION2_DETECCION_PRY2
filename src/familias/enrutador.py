"""Fusión de los especialistas: P(clase) por regresión logística y P(familia) como suma de sus clases."""
# ─────────────────────────────  Imports  ─────────────────────────────
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.familias.especialistas import Especialista, log_probabilidades
from src.familias.vistas import VISTAS, vistas


# ─────────────────────────────  Fusión  ──────────────────────────────
class Fusion:
    def __init__(self, clases, familias_de_clase, C=1.0):
        self.clases = list(clases)
        self.familia_de = dict(familias_de_clase)
        self.familias = sorted(set(self.familia_de.values()))
        self.C = C
        self.modelo = None

    def _rasgos(self, logp):
        return np.concatenate([logp[v] for v in VISTAS], axis=1)

    def ajustar(self, logp, y):
        self.modelo = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=3000))
        self.modelo.fit(self._rasgos(logp), y)
        return self

    def probabilidades(self, logp):
        p = np.zeros((len(logp[VISTAS[0]]), len(self.clases)))
        p[:, self.modelo[-1].classes_] = self.modelo.predict_proba(self._rasgos(logp))
        return p

    def por_familia(self, p):
        return np.stack([p[:, [i for i, c in enumerate(self.clases) if self.familia_de[c] == f]].sum(1)
                         for f in self.familias], axis=1)


# ────────────────────────────  Enrutador  ────────────────────────────
class Enrutador:
    """Especialistas por vista más la fusión. Analiza una señal I/Q sin saber su tasa de símbolo."""

    def __init__(self, especialistas, fusion):
        self.especialistas = especialistas
        self.fusion = fusion

    def log_probabilidades(self, x_por_vista):
        return {v: log_probabilidades(self.especialistas[v], x_por_vista[v]) for v in VISTAS}

    def analizar(self, iq):
        v, banda = vistas(iq)
        logp = self.log_probabilidades({k: a[None] for k, a in v.items()})
        p = self.fusion.probabilidades(logp)
        familias = self.fusion.por_familia(p)[0]
        votos = {k: {c: float(np.exp(x)) for c, x in zip(self.fusion.clases, logp[k][0])} for k in VISTAS}
        return {"clases": dict(zip(self.fusion.clases, p[0].tolist())),
                "familias": dict(zip(self.fusion.familias, familias.tolist())),
                "especialistas": votos, "banda": banda}

    def guardar(self, carpeta):
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        for k, m in self.especialistas.items():
            torch.save({"config": m.config, "estado": m.cpu().state_dict()}, carpeta / f"especialista_{k}.pt")
        joblib.dump(self.fusion, carpeta / "fusion.joblib")

    @classmethod
    def cargar(cls, carpeta, disp="cpu"):
        carpeta = Path(carpeta)
        especialistas = {}
        for k in VISTAS:
            d = torch.load(carpeta / f"especialista_{k}.pt", map_location=disp, weights_only=True)
            m = Especialista(**d["config"])
            m.load_state_dict(d["estado"])
            especialistas[k] = m.to(disp).eval()
        return cls(especialistas, joblib.load(carpeta / "fusion.joblib"))
