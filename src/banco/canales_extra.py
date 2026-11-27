"""Canales e interferencias adicionales del banco.

CONTRATO (unidad U1, cerrado):

Toda función de canal tiene la firma ``f(x, sps, rng) -> (y, parametros)``:
- ``x``: np.ndarray complejo 1-D (cualquier dtype complejo), señal transmitida con ``sps`` muestras por símbolo.
- ``sps``: float > 1, muestras por símbolo.
- ``rng``: np.random.Generator; toda aleatoriedad sale de él (misma semilla -> misma salida).
- ``y``: np.ndarray complex128 con ``len(y) == len(x)``, todo finito.
- ``parametros``: dict con valores float, int o str (serializable a JSON) que describen lo sorteado.
- No normalizar la potencia de ``y`` salvo donde se indica. No agregar ruido térmico (lo agrega ``sintetizar``).
- Si ``len(x) == 0`` devolver ``(np.zeros(0, complex), {})``.

Dependencias permitidas (lista cerrada): numpy, scipy.interpolate.CubicSpline, scipy.signal.fftconvolve,
src.waveform.rrc_filter, y de src.banco.canales: Canal, CANALES, potencia, _jakes, _sumar_rayos.

Funciones auxiliares:
- ``_ganancias_jakes(n, fd_por_muestra, k, rng)`` -> lista de k arrays complejos de largo n, cada uno ``_jakes(n, fd_por_muestra, rng)``.
- ``_senal_modulada(n, tasa, rng)`` -> (array complejo de largo n con potencia 1, str con "QPSK" o "16QAM").
  Elige "QPSK" o "16QAM" con igual probabilidad. QPSK = exp(j(pi/4 + k pi/2)); 16QAM = rejilla {-3,-1,1,3}^2 / sqrt(10).
  Genera ``m = ceil(n * tasa) + 24`` símbolos, los sube a 8 muestras/símbolo (tren de impulsos), filtra con
  ``rrc_filter(beta=0.35, sps=8, span=10)`` por ``fftconvolve`` modo "full" recortado desde ``(len(h)-1)//2``,
  y remuestrea con CubicSpline en las posiciones ``t = 8*12 + np.arange(n) * 8 * tasa`` (tasa en símbolos por muestra).
  Normaliza a potencia 1.
- ``_escalar_sir(x, i, sir_db)`` -> i escalada para que potencia(x)/potencia(i) = 10**(sir_db/10); si potencia(i) es 0 devuelve i.

Perfiles LTE (3GPP TS 36.104, anexo B.2), retardos en ns y potencias en dB:
  EPA: [0, 30, 70, 90, 110, 190, 410], [0, -1, -2, -3, -8, -17.2, -20.8], Doppler 5 Hz
  EVA: [0, 30, 150, 310, 370, 710, 1090, 1730, 2510], [0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9], Doppler 70 Hz
  ETU: [0, 50, 120, 200, 230, 500, 1600, 2300, 5000], [-1, -1, -1, 0, 0, 0, -3, -5, -7], Doppler 300 Hz
``_lte(nombre)`` devuelve una función de canal que: sortea la tasa de símbolo Rs log-uniforme en [2e5, 2e6] Bd;
retardos en muestras = retardo_ns * 1e-9 * Rs * sps; magnitudes = 10**(dB/20) normalizadas a energía 1;
ganancias = _ganancias_jakes(len(x), doppler_hz / Rs / sps, n_rayos, rng); y = _sumar_rayos(x, retardos, magnitudes, ganancias).
parametros: {"perfil": nombre, "tasa_bd": Rs, "doppler_por_simbolo": doppler_hz / Rs}.

Canales:
- ``nakagami(x, sps, rng)``: m ~ U(0.5, 4); bloque B ~ U(20, 200) símbolos; nodos cada B*sps muestras cubriendo
  todo x (al menos 2 nodos); en cada nodo potencia P ~ Gamma(forma=m, escala=1/m) y fase phi_k = phi_{k-1} + N(0, 0.3)
  con phi_0 ~ U(0, 2pi); ganancia compleja sqrt(P) exp(j phi) interpolada con CubicSpline (parte real e imaginaria
  por separado) sobre el índice de muestra; y = x * g. parametros {"m", "bloque_simbolos"}.
- ``sombra(x, sps, rng)``: sigma_db ~ U(4, 10); correlación L ~ U(100, 1000) símbolos; nodos cada L/4 símbolos
  (al menos 4) con valores N(0, 1); interpolar con CubicSpline; restar la media y dividir por el desvío del resultado;
  s = sigma_db * curva; y = x * 10**(s/20). parametros {"sigma_db", "correlacion_simbolos"}.
- ``dos_rayos(x, sps, rng)``: d ~ U(0.1, 2.0) símbolos; gamma ~ U(0.3, 0.95); fase ~ U(0, 2pi);
  y = _sumar_rayos(x, [0, d*sps], [1, gamma]/norma, [1, -exp(j fase)]) con norma = sqrt(1 + gamma^2), ganancias como
  arrays constantes de largo len(x). parametros {"retardo_simbolos", "reflexion"}.
- ``ecos_largos(x, sps, rng)``: k ~ entero en {2, 3}; retardos U(4, 20) símbolos ordenados; magnitudes U(0.2, 0.6);
  fases U(0, 2pi); rayo principal en 0 con magnitud 1 y fase 0; magnitudes normalizadas a energía 1; ganancias
  constantes. parametros {"ecos": k, "retardo_max_simbolos"}.
- ``watterson(x, sps, rng)``: dos trayectos; retardo del segundo U(0.5, 3) símbolos; potencia del segundo -U(0, 6) dB;
  dispersión Doppler sigma ~ U(1e-4, 3e-3) ciclos/símbolo; corrimiento por trayecto U(-5e-4, 5e-4) ciclos/símbolo.
  Ganancia de cada trayecto: ruido complejo blanco de largo len(x) filtrado en frecuencia por exp(-f^2 / (2 s^2))
  con f = np.fft.fftfreq(len(x)) y s = sigma / sps, normalizado a potencia 1, multiplicado por
  exp(j 2pi corrimiento/sps n). y = _sumar_rayos con magnitudes normalizadas a energía 1.
  parametros {"retardo_simbolos", "dispersion_por_simbolo", "potencia_segundo_db"}.
- ``int_adyacente(x, sps, rng)``: rho ~ U(0.5, 2.0); tasa_i = rho / sps; mitad_x = 1.35/(2 sps); mitad_i = 1.35 tasa_i / 2;
  desvio = signo * U(0.6, 1.0) * (mitad_x + mitad_i) con signo +-1 equiprobable; i = _senal_modulada(len(x), tasa_i, rng)
  * exp(j 2pi desvio n); sir ~ U(0, 20) dB; y = x + _escalar_sir(x, i, sir). parametros {"sir_db", "desvio", "modulacion", "razon_tasa"}.
- ``int_cocanal(x, sps, rng)``: igual que int_adyacente pero desvio ~ U(-0.1, 0.1) / sps y sir ~ U(10, 25) dB.
- ``int_barrido(x, sps, rng)``: ancho b = U(0.5, 1.5) / sps ciclos/muestra; periodo P = U(50, 500) * sps muestras;
  frecuencia instantánea f(n) = -b/2 + b * ((n mod P) / P); fase = 2pi * cumsum(f); i = exp(j fase); sir ~ U(5, 20) dB.
  parametros {"sir_db", "ancho", "periodo_simbolos"}.
- ``int_pulsos(x, sps, rng)``: PRI = U(50, 400) símbolos; ciclo de trabajo c = U(0.01, 0.1); el pulso ocupa
  max(1, round(c * PRI * sps)) muestras al inicio de cada periodo (inicio del primero U(0, PRI*sps)); contenido
  tono exp(j 2pi f0 n) con f0 ~ U(-0.5, 0.5) / sps; sir_pico ~ U(-10, 10) dB: amplitud del pulso
  sqrt(potencia(x) / 10**(sir_pico/10)); fuera del pulso i = 0. parametros {"sir_pico_db", "pri_simbolos", "ciclo"}.

Registro:
- ``FUNCIONES_EXTRA``: dict nombre -> función, con las claves exactas
  "lte_epa", "lte_eva", "lte_etu", "nakagami", "sombra", "dos_rayos", "ecos_largos", "watterson",
  "int_adyacente", "int_cocanal", "int_barrido", "int_pulsos".
- ``CANALES_EXTRA = {k: Canal(antes=f) for k, f in FUNCIONES_EXTRA.items()}``
- ``TODOS_CANALES = {**CANALES, **CANALES_EXTRA}``

Criterio de aceptación: para cada canal y sps en (2.5, 8, 32), con x QPSK RRC de 4000 muestras y rng con
semilla fija: largo igual, finito, potencia de y entre 0.01 y 100 veces la de x, misma semilla -> misma salida,
parametros serializable con json.dumps.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from src.banco.canales import CANALES, Canal, _jakes, _sumar_rayos, potencia
from src.waveform import rrc_filter

PERFILES_LTE = {
    "epa": ([0, 30, 70, 90, 110, 190, 410], [0, -1, -2, -3, -8, -17.2, -20.8], 5.0),
    "eva": ([0, 30, 150, 310, 370, 710, 1090, 1730, 2510], [0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9], 70.0),
    "etu": ([0, 50, 120, 200, 230, 500, 1600, 2300, 5000], [-1, -1, -1, 0, 0, 0, -3, -5, -7], 300.0),
}


def _ganancias_jakes(n, fd_por_muestra, k, rng):
    return [_jakes(n, fd_por_muestra, rng) for _ in range(k)]


def _senal_modulada(n, tasa, rng):
    mod = str(rng.choice(["QPSK", "16QAM"]))
    if n == 0:
        return np.zeros(0, complex), mod
    m = int(np.ceil(n * tasa)) + 24
    if mod == "QPSK":
        simbolos = np.exp(1j * (np.pi / 4 + rng.integers(0, 4, size=m) * (np.pi / 2)))
    else:
        niveles = np.array([-3.0, -1.0, 1.0, 3.0])
        simbolos = (rng.choice(niveles, size=m) + 1j * rng.choice(niveles, size=m)) / np.sqrt(10.0)
    tren = np.zeros(m * 8, dtype=np.complex128)
    tren[::8] = simbolos
    h = rrc_filter(beta=0.35, sps=8, span=10)
    filtrada = fftconvolve(tren, h, mode="full")[(len(h) - 1) // 2 :]
    t = 8 * 12 + np.arange(n, dtype=float) * 8 * tasa
    s = np.asarray(CubicSpline(np.arange(len(filtrada), dtype=float), filtrada)(t), dtype=np.complex128)
    p = potencia(s)
    if p > 0:
        s = s / np.sqrt(p)
    return s, mod


def _escalar_sir(x, i, sir_db):
    if len(x) == 0 or len(i) == 0:
        return i
    pi = potencia(i)
    if pi == 0:
        return i
    return i * np.sqrt(potencia(x) / pi / (10.0 ** (sir_db / 10.0)))


def _lte(nombre):
    retardos_ns, potencias_db, doppler_hz = PERFILES_LTE[nombre]
    mags = 10.0 ** (np.asarray(potencias_db, dtype=float) / 20.0)
    mags = mags / np.sqrt(np.sum(mags ** 2))
    retardos_base = np.asarray(retardos_ns, dtype=float) * 1e-9

    def canal(x, sps, rng):
        if len(x) == 0:
            return np.zeros(0, complex), {}
        x = np.asarray(x, dtype=np.complex128)
        rs = float(np.exp(rng.uniform(np.log(2e5), np.log(2e6))))
        retardos = retardos_base * rs * sps
        ganancias = _ganancias_jakes(len(x), doppler_hz / rs / sps, len(retardos), rng)
        y = _sumar_rayos(x, retardos, mags, ganancias)
        return y, {"perfil": nombre, "tasa_bd": rs, "doppler_por_simbolo": doppler_hz / rs}

    return canal


def nakagami(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    m = float(rng.uniform(0.5, 4.0))
    bloque = float(rng.uniform(20.0, 200.0))
    paso = bloque * sps
    n_nodos = max(2, int(np.ceil((n - 1) / paso)) + 1)
    nodos = np.arange(n_nodos, dtype=float) * paso
    p = rng.gamma(shape=m, scale=1.0 / m, size=n_nodos)
    phi = np.cumsum(np.r_[rng.uniform(0.0, 2.0 * np.pi), rng.normal(0.0, 0.3, size=n_nodos - 1)])
    gn = np.sqrt(p) * np.exp(1j * phi)
    idx = np.arange(n, dtype=float)
    g = CubicSpline(nodos, gn.real)(idx) + 1j * CubicSpline(nodos, gn.imag)(idx)
    return x * g, {"m": m, "bloque_simbolos": bloque}


def sombra(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    sigma_db = float(rng.uniform(4.0, 10.0))
    corr = float(rng.uniform(100.0, 1000.0))
    paso = (corr / 4.0) * sps
    n_nodos = max(4, int(np.ceil((n - 1) / paso)) + 1)
    nodos = np.arange(n_nodos, dtype=float) * paso
    valores = rng.normal(0.0, 1.0, size=n_nodos)
    curva = CubicSpline(nodos, valores)(np.arange(n, dtype=float))
    curva = curva - np.mean(curva)
    std = float(np.std(curva))
    if std > 0:
        curva = curva / std
    s = sigma_db * curva
    return x * (10.0 ** (s / 20.0)), {"sigma_db": sigma_db, "correlacion_simbolos": corr}


def dos_rayos(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    d = float(rng.uniform(0.1, 2.0))
    gamma = float(rng.uniform(0.3, 0.95))
    fase = float(rng.uniform(0.0, 2.0 * np.pi))
    norma = np.sqrt(1.0 + gamma ** 2)
    magnitudes = np.array([1.0, gamma], dtype=float) / norma
    ganancias = [np.ones(n, dtype=np.complex128), np.full(n, -np.exp(1j * fase), dtype=np.complex128)]
    y = _sumar_rayos(x, [0.0, d * sps], magnitudes, ganancias)
    return y, {"retardo_simbolos": d, "reflexion": gamma}


def ecos_largos(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    k = int(rng.integers(2, 4))
    ret_ecos = np.sort(rng.uniform(4.0, 20.0, size=k))
    retardos = np.r_[0.0, ret_ecos] * sps
    magnitudes = np.r_[1.0, rng.uniform(0.2, 0.6, size=k)]
    magnitudes = magnitudes / np.sqrt(np.sum(magnitudes ** 2))
    fases = rng.uniform(0.0, 2.0 * np.pi, size=k)
    ganancias = [np.ones(n, dtype=np.complex128)] + [np.full(n, np.exp(1j * f), dtype=np.complex128) for f in fases]
    y = _sumar_rayos(x, retardos, magnitudes, ganancias)
    return y, {"ecos": k, "retardo_max_simbolos": float(ret_ecos[-1])}


def watterson(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    retardo = float(rng.uniform(0.5, 3.0))
    potencia_segundo_db = float(-rng.uniform(0.0, 6.0))
    sigma = float(rng.uniform(1e-4, 3e-3))
    corrimientos = rng.uniform(-5e-4, 5e-4, size=2)
    f = np.fft.fftfreq(n)
    filtro = np.exp(-(f ** 2) / (2.0 * (sigma / sps) ** 2))
    idx = np.arange(n, dtype=float)
    ganancias = []
    for corr in corrimientos:
        ruido = rng.normal(size=n) + 1j * rng.normal(size=n)
        g = np.fft.ifft(np.fft.fft(ruido) * filtro)
        pg = potencia(g)
        if pg > 0:
            g = g / np.sqrt(pg)
        ganancias.append(g * np.exp(2j * np.pi * (corr / sps) * idx))
    magnitudes = np.array([1.0, 10.0 ** (potencia_segundo_db / 20.0)], dtype=float)
    magnitudes = magnitudes / np.sqrt(np.sum(magnitudes ** 2))
    y = _sumar_rayos(x, [0.0, retardo * sps], magnitudes, ganancias)
    return y, {
        "retardo_simbolos": retardo,
        "dispersion_por_simbolo": sigma,
        "potencia_segundo_db": potencia_segundo_db,
    }


def _aplicar_interferencia_modulada(x, sps, rng, rho, desvio, sir):
    n = len(x)
    s_i, mod = _senal_modulada(n, rho / sps, rng)
    i = s_i * np.exp(2j * np.pi * desvio * np.arange(n, dtype=float))
    y = x + _escalar_sir(x, i, sir)
    return y, {"sir_db": sir, "desvio": desvio, "modulacion": mod, "razon_tasa": rho}


def int_adyacente(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    rho = float(rng.uniform(0.5, 2.0))
    tasa_i = rho / sps
    mitad_x = 1.35 / (2.0 * sps)
    mitad_i = 1.35 * tasa_i / 2.0
    signo = float(rng.choice([-1.0, 1.0]))
    desvio = float(signo * rng.uniform(0.6, 1.0) * (mitad_x + mitad_i))
    sir = float(rng.uniform(0.0, 20.0))
    return _aplicar_interferencia_modulada(x, sps, rng, rho, desvio, sir)


def int_cocanal(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    rho = float(rng.uniform(0.5, 2.0))
    desvio = float(rng.uniform(-0.1, 0.1) / sps)
    sir = float(rng.uniform(10.0, 25.0))
    return _aplicar_interferencia_modulada(x, sps, rng, rho, desvio, sir)


def int_barrido(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x), dtype=float)
    ancho = float(rng.uniform(0.5, 1.5) / sps)
    periodo_simbolos = float(rng.uniform(50.0, 500.0))
    periodo = periodo_simbolos * sps
    f_inst = -ancho / 2.0 + ancho * ((n % periodo) / periodo)
    i = np.exp(1j * (2.0 * np.pi * np.cumsum(f_inst)))
    sir = float(rng.uniform(5.0, 20.0))
    y = x + _escalar_sir(x, i, sir)
    return y, {"sir_db": sir, "ancho": ancho, "periodo_simbolos": periodo_simbolos}


def int_pulsos(x, sps, rng):
    if len(x) == 0:
        return np.zeros(0, complex), {}
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x), dtype=float)
    pri_simbolos = float(rng.uniform(50.0, 400.0))
    ciclo = float(rng.uniform(0.01, 0.1))
    pri_muestras = pri_simbolos * sps
    ancho_pulso = max(1, round(ciclo * pri_muestras))
    inicio = float(rng.uniform(0.0, pri_muestras))
    f0 = float(rng.uniform(-0.5, 0.5) / sps)
    sir_pico = float(rng.uniform(-10.0, 10.0))
    amplitud = np.sqrt(potencia(x) / (10.0 ** (sir_pico / 10.0)))
    pulso = ((n - inicio) % pri_muestras) < ancho_pulso
    i = np.where(pulso, amplitud * np.exp(2j * np.pi * f0 * n), 0.0 + 0.0j)
    return x + i, {"sir_pico_db": sir_pico, "pri_simbolos": pri_simbolos, "ciclo": ciclo}


FUNCIONES_EXTRA = {
    "lte_epa": _lte("epa"),
    "lte_eva": _lte("eva"),
    "lte_etu": _lte("etu"),
    "nakagami": nakagami,
    "sombra": sombra,
    "dos_rayos": dos_rayos,
    "ecos_largos": ecos_largos,
    "watterson": watterson,
    "int_adyacente": int_adyacente,
    "int_cocanal": int_cocanal,
    "int_barrido": int_barrido,
    "int_pulsos": int_pulsos,
}
CANALES_EXTRA = {k: Canal(antes=f) for k, f in FUNCIONES_EXTRA.items()}
TODOS_CANALES = {**CANALES, **CANALES_EXTRA}
CANALES_CON_ECOS = ("proyecto", "mp_corto", "mp_largo", "tdl_rayleigh", "tdl_rician", "gr_multitrayecto",
                    "gr_selectivo", "gr_dinamico", "lte_epa", "lte_eva", "lte_etu", "dos_rayos", "ecos_largos",
                    "watterson")  # los que obedecen al regulador de ISI (src/banco/isi.py)
