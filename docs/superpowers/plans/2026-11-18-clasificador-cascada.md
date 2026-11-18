# Clasificador en cascada: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** clasificar archivos I/Q de generadores no vistos, con probabilidades y rechazo de modulaciones desconocidas.

**Architecture:** normalizador clásico → recuperador (red o cadena clásica) → autofoco → descriptor contrastivo → decisor PCA + GMM. Paquete nuevo `src/cascada/`; el monitor actual sigue igual salvo una opción nueva.

**Tech Stack:** Python 3.12, NumPy, SciPy, scikit-learn, PyTorch CPU, joblib, GNU Radio 3.10.12 (solo para generar prueba).

**Spec:** `docs/superpowers/specs/2026-11-18-clasificador-cascada-design.md`

## Global Constraints

- Solo CPU: Ryzen 9 5900HS, 8 núcleos, 15 GB. La RTX está en modo Eco y no se puede activar sin el usuario.
- Formato canónico: 8 muestras por símbolo. La cascada no requiere `fs_hz`.
- Catálogo: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 4PAM.
- Desconocidas de validación: 6PSK, 24QAM, HEX19, ANILLOS12.
- Desconocidas de prueba: 32QAM, 128QAM, 16APSK, 32APSK, 12PSK, 8QAM_ESTRELLA, ARB_A, ARB_B.
- Salidas: nombre del catálogo, `DESCONOCIDA` o `INDETERMINADA`.
- Nada de ajustar con el conjunto de prueba final. Se ejecuta una sola vez.
- `pytest` no debe empeorar respecto a la línea base: 269 aprobadas, 7 fallidas por artefactos ausentes, 2 omitidas.
- Sin push. Commits solo en la rama local `cascada`.

## Cambios respecto a la especificación por falta de GPU

| Especificación | Plan | Motivo |
|---|---|---|
| Cabeza de temporización en R | Oerder-Meyr sobre la salida limpia | Clásico, sin coste de entrenamiento |
| Evidencia sumada por bloque | Densidad acumulada de todos los símbolos del archivo, con segmentos solo si es muy largo | Acumula evidencia sin suponer independencia entre bloques |
| Hasta 128 puntos en entrenamiento, redes medianas | Igual en geometría; redes de menos de 300 k parámetros | Entrenar en CPU |

## Review Focus

1. Ruido puro: no debe nombrar una modulación del catálogo con confianza alta. Prueba en la tarea 6.
2. Escala arbitraria de amplitud: ×0,01 y ×100 dan la misma decisión. Prueba en la tarea 6.
3. Tasa no entera (por ejemplo 5,3 muestras por símbolo): se normaliza. Prueba en la tarea 2.
4. Tono puro o FSK: no falla y no devuelve una modulación del catálogo con confianza alta. Prueba en la tarea 6.
5. Archivo corto, de menos de 256 muestras: devuelve `INDETERMINADA`, sin excepción. Prueba en la tarea 6.

---

### Task 1: Geometrías y síntesis

**Files:** Create `src/cascada/__init__.py`, `src/cascada/geometria.py`, `src/cascada/sintesis.py`. Test `tests/test_cascada_datos.py`.

**Interfaces (Produces):**
- `catalogo() -> dict[str, ndarray]`, `desconocidas_validacion()`, `desconocidas_prueba()`: potencia media 1.
- `geometria_aleatoria(rng) -> ndarray`, que rechaza lo parecido a las desconocidas de prueba y validación.
- `distancia_geometrica(a, b) -> float`, invariante a fase.
- `contar_anillos(puntos) -> int`.
- `Condicion` (dataclass) y `condicion_aleatoria(rng, snr_rango)`.
- `transmitir(puntos, n_simbolos, cond, rng) -> dict` con `iq`, `limpia`, `simbolos`.

- [ ] Pruebas: potencia unitaria, número de puntos del catálogo, rechazo de desconocidas, SNR medida ±0,5 dB, `limpia` sin ruido alineada con `iq` cuando el canal es ideal.
- [ ] Implementar y pasar. Commit.

### Task 2: Normalizador

**Files:** Create `src/cascada/normalizador.py`. Test `tests/test_cascada_normalizador.py`.

**Produces:** `Hipotesis(iq, sps_origen, puntaje)`; `normalizar(iq, sps=None, max_hipotesis=3) -> list[Hipotesis]`.

- [ ] Pruebas: recupera sps 4, 8, 5,3 y 13 con error < 1 % a 15 dB; CFO grueso reducido; escala invariante; `sps` forzado.
- [ ] Implementar y pasar. Commit.

### Task 3: Cadena clásica y autofoco

**Files:** Create `src/cascada/clasica.py`, `src/cascada/autofoco.py`. Test `tests/test_cascada_simbolos.py`.

**Produces:** `tomar_simbolos(onda, sps=8) -> ndarray` (Oerder-Meyr con seguimiento); `cma(simbolos_2sps...)`; `recuperar_clasico(iq8) -> onda`; `autofoco(simbolos) -> (simbolos, df, nitidez)`.

- [ ] Pruebas: 16QAM a 25 dB con CFO residual queda con EVM < 15 %; geometría arbitraria también enfoca; multitrayecto leve mejora con CMA.
- [ ] Implementar y pasar. Commit.
- [ ] **H0:** `experiments/exp_c0_viabilidad.py`, con salida en `results/cascada/h0.json`. Criterio de la especificación.

### Task 4: Recuperador

**Files:** Create `src/cascada/recuperador.py`, `experiments/exp_c1_recuperador.py`. Test `tests/test_cascada_recuperador.py`.

**Produces:** `Recuperador(nn.Module)`, entrada y salida `(B,2,N)`; `perdida_fase_invariante(y, t)`; `aplicar_recuperador(modelo, iq8) -> ndarray`; `cargar_recuperador(ruta)`.

- [ ] Pruebas: forma, longitud arbitraria, pérdida invariante a fase, solape y suma sin discontinuidad.
- [ ] Implementar. Entrenar con tope de 60 min en segundo plano.
- [ ] **H2:** comparar EVM contra la cadena clásica con canales no vistos y escribir `results/cascada/h2.json` con la cadena elegida. Commit.

### Task 5: Descriptor

**Files:** Create `src/cascada/descriptor.py`, `experiments/exp_c2_descriptor.py`. Test `tests/test_cascada_descriptor.py`.

**Produces:** `imagen_densidad(simbolos) -> ndarray (2,64,64)`; `Descriptor(nn.Module)` con `forward -> (huella, log2M, anillos)`; `perdida_contrastiva(z1, z2, t=0.1)`; `huella(modelo, imagenes)`.

- [ ] Pruebas: imagen acotada e invariante a escala, huella de norma 1, pérdida menor con pares iguales.
- [ ] Implementar. Generar vistas con la cadena elegida y entrenar con tope de 60 min. Commit.

### Task 6: Decisor y tubería

**Files:** Create `src/cascada/decisor.py`, `src/cascada/pipeline.py`, `src/cascada/lote.py`, `experiments/exp_c3_inscripcion.py`. Modify `src/monitor.py` (`--motor`, `--sps`). Test `tests/test_cascada_pipeline.py`.

**Produces:** `Decisor.ajustar(...)`, `.probabilidades(z)`, `.decidir(z)`, `.inscribir(nombre, z)`; `analizar_cascada(iq, paquete, sps=None) -> dict`; `cargar_paquete(ruta)`.

- [ ] Pruebas: decisor con datos sintéticos, inscripción sin reentrenar, y los cinco puntos de Review Focus.
- [ ] Implementar, inscribir, calibrar en validación. Commit.

### Task 7: Evaluación final

**Files:** Create `experiments/gen_gnuradio_cascada.py`, `experiments/gen_independiente.py`, `experiments/exp_c4_final.py`.

- [ ] Generar GNU Radio y generador independiente; intentar TorchSig en un entorno aparte.
- [ ] Comparar con SVM reentrenado si cabe en tiempo.
- [ ] Ejecutar una vez y guardar en `results/cascada/final/`.
- [ ] Inscribir 16APSK y medir en datos nuevos. Commit.

### Task 8: Informe

**Files:** Create `docs/CASCADA_RESULTADOS.md`. Modify `README.md`.

- [ ] Resultados, incluidos los negativos, y cómo reproducirlos. `pytest` completo. Commit.
