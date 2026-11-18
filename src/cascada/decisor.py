"""Etapa 5: PCA, mezclas gaussianas y decisión con hipótesis de desconocida."""
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DESCONOCIDA = "DESCONOCIDA"
INDETERMINADA = "INDETERMINADA"
N_CALIDAD = 3


class Decisor:
    def __init__(self, dim_max=32, componentes=3, componentes_fondo=24, semilla=0):
        self.dim_max = dim_max
        self.componentes = componentes
        self.componentes_fondo = componentes_fondo
        self.semilla = semilla
        self.modelos = {}
        self.temperatura = 1.0
        self.umbral = 0.5
        self.puerta = None
        self.calidad_tipica = np.zeros(N_CALIDAD)

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

    @staticmethod
    def _softmax(a):
        return np.exp(a - logsumexp(a, axis=1, keepdims=True))

    def _cerrado(self, l, temperatura):
        return self._softmax(l[:, :-1] / temperatura)

    def _rasgos(self, l, calidad):
        """Rasgos que no dependen de cuántas modulaciones haya inscritas."""
        orden = np.sort(l[:, :-1], axis=1)
        mejor = orden[:, -1]
        segundo = orden[:, -2] if orden.shape[1] > 1 else mejor
        p = self._cerrado(l, self.temperatura)
        h = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
        return np.c_[mejor - l[:, -1], mejor - segundo, h, mejor, l[:, -1], calidad]

    def _calidad(self, n, calidad):
        if calidad is None:
            return np.tile(self.calidad_tipica, (n, 1))
        return np.nan_to_num(np.atleast_2d(np.asarray(calidad, dtype=float)), posinf=100.0, neginf=-100.0)

    def probabilidades(self, z, calidad=None):
        l = self._verosimilitudes(z)
        if self.puerta is None:
            return self._softmax(l / self.temperatura)
        p_fondo = self.puerta.predict_proba(self._rasgos(l, self._calidad(len(l), calidad)))[:, 1]
        return np.c_[self._cerrado(l, self.temperatura) * (1 - p_fondo)[:, None], p_fondo]

    def calibrar(self, z, etiquetas, calidad=None, exactitud_objetivo=0.85, C=1.0):
        """Ajusta temperatura del catálogo, puerta de desconocidas y umbral."""
        l = self._verosimilitudes(z)
        nombres = self.nombres
        y = np.array([nombres.index(e) for e in etiquetas])
        fondo = len(nombres) - 1
        conocidas = y != fondo
        filas = np.arange(conocidas.sum())
        mejor = (np.inf, 1.0)
        for t in np.geomspace(0.5, 100, 40):
            nll = -np.mean(np.log(np.maximum(self._cerrado(l[conocidas], t)[filas, y[conocidas]], 1e-12)))
            if nll < mejor[0]:
                mejor = (nll, float(t))
        self.temperatura = mejor[1]
        q = self._calidad(len(l), calidad) if calidad is not None else np.zeros((len(l), N_CALIDAD))
        self.calidad_tipica = np.median(q, axis=0)
        self.puerta = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000)).fit(
            self._rasgos(l, q), (y == fondo).astype(int))
        p = self.probabilidades(z, q)
        confianza, acierto = p.max(1), p.argmax(1) == y
        self.umbral = 0.95
        for u in np.linspace(0.3, 0.95, 66):
            aceptados = confianza >= u
            if aceptados.mean() >= 0.05 and acierto[aceptados].mean() >= exactitud_objetivo:
                self.umbral = float(u)
                break
        return self

    def decidir(self, z, calidad=None):
        p = self.probabilidades(z, calidad)[0]
        k = int(p.argmax())
        nombre = self.nombres[k]
        confianza = float(p[k])
        if confianza < self.umbral:
            nombre = INDETERMINADA
        return {"modulacion": nombre, "confianza": confianza,
                "probabilidades": {n: float(v) for n, v in zip(self.nombres, p)}}

    def proyectar(self, z):
        return self.pca.transform(np.atleast_2d(z))
