"""Etapa 5: PCA, mezclas gaussianas y decisión con hipótesis de desconocida."""
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

DESCONOCIDA = "DESCONOCIDA"
INDETERMINADA = "INDETERMINADA"


class Decisor:
    def __init__(self, dim_max=32, componentes=3, componentes_fondo=24, semilla=0):
        self.dim_max = dim_max
        self.componentes = componentes
        self.componentes_fondo = componentes_fondo
        self.semilla = semilla
        self.modelos = {}
        self.temperatura = 1.0
        self.log_previa_fondo = 0.0
        self.umbral = 0.5

    def _mezcla(self, z, k):
        k = max(1, min(k, len(z) // (self.pca.n_components_ + 2)))
        return GaussianMixture(k, covariance_type="full", reg_covar=1e-2, n_init=1, max_iter=120,
                               random_state=self.semilla).fit(z)

    def ajustar(self, fondo, catalogo):
        todas = np.concatenate([fondo, *catalogo.values()])
        n = min(self.dim_max, todas.shape[1], len(todas) - 1)
        self.pca = PCA(n_components=n, whiten=True, random_state=self.semilla).fit(todas)
        acumulada = np.cumsum(self.pca.explained_variance_ratio_)
        conservar = int(min(n, np.searchsorted(acumulada, 0.95) + 1))
        self.pca = PCA(n_components=conservar, whiten=True, random_state=self.semilla).fit(todas)
        self.fondo = self._mezcla(self.pca.transform(fondo), self.componentes_fondo)
        self.modelos = {}
        for nombre, z in catalogo.items():
            self.inscribir(nombre, z)
        return self

    def inscribir(self, nombre, z):
        """Añade o reemplaza una modulación sin tocar las redes."""
        self.modelos[nombre] = self._mezcla(self.pca.transform(z), self.componentes)
        return self

    @property
    def nombres(self):
        return [*self.modelos, DESCONOCIDA]

    def _verosimilitudes(self, z):
        v = self.pca.transform(np.atleast_2d(z))
        return np.column_stack([*(m.score_samples(v) for m in self.modelos.values()),
                                self.fondo.score_samples(v)])

    def _posterior(self, l, temperatura, previa_fondo):
        previa = np.zeros(l.shape[1])
        previa[-1] = previa_fondo
        a = l / temperatura + previa
        return np.exp(a - logsumexp(a, axis=1, keepdims=True))

    def probabilidades(self, z):
        return self._posterior(self._verosimilitudes(z), self.temperatura, self.log_previa_fondo)

    def calibrar(self, z, etiquetas, exactitud_objetivo=0.85):
        """Ajusta temperatura, previa del fondo y umbral con datos de validación."""
        l = self._verosimilitudes(z)
        nombres = self.nombres
        y = np.array([nombres.index(e) for e in etiquetas])
        filas = np.arange(len(y))
        mejor = (np.inf, 1.0, 0.0)
        for t in np.geomspace(1, 200, 40):
            for previa in np.linspace(-6, 6, 25):
                p = self._posterior(l, t, previa)
                nll = -np.mean(np.log(np.maximum(p[filas, y], 1e-12)))
                if nll < mejor[0]:
                    mejor = (nll, float(t), float(previa))
        _, self.temperatura, self.log_previa_fondo = mejor
        p = self.probabilidades(z)
        confianza, acierto = p.max(1), p.argmax(1) == y
        self.umbral = 0.95
        for u in np.linspace(0.3, 0.95, 66):
            aceptados = confianza >= u
            if aceptados.mean() >= 0.05 and acierto[aceptados].mean() >= exactitud_objetivo:
                self.umbral = float(u)
                break
        return self

    def decidir(self, z):
        p = self.probabilidades(z)[0]
        k = int(p.argmax())
        nombre = self.nombres[k]
        confianza = float(p[k])
        if confianza < self.umbral:
            nombre = INDETERMINADA
        return {"modulacion": nombre, "confianza": confianza,
                "probabilidades": {n: float(v) for n, v in zip(self.nombres, p)}}

    def proyectar(self, z):
        return self.pca.transform(np.atleast_2d(z))
