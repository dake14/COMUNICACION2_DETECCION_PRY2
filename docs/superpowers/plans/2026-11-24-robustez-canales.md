# Robustez a canales, interferencias y sobremuestreo

Trabajo autónomo pedido el 2026-11-24 con `/superteam` (ejecutores agy `gemini-4-hydrogen-high` y
`gemini-4-neon-high`). El usuario no está: modo **por lotes**, sin commits (respaldo del código en
`..\respaldos\proyectocom22-2026-11-24-*`).

## Diagnóstico de partida (prueba parcial exp_c6, 6 720 archivos)

| Canal | Exactitud conocidas | Tasa bien estimada |
|---|---|---|
| gr_dinamico, gr_multitrayecto | 33 % | 80–87 % |
| mp_largo, tdl_rayleigh, tdl_rician, gr_selectivo | 37 % | 88–91 % |
| mp_corto | 58 % | 83 % |
| interferencia | 83 % | 97 % |
| ideal | 92 % | 99 % |

Causa: el recuperador (`exp_c1`) solo vio multitrayecto estático (`sintesis.canal_aleatorio`), el
ecualizador usa pesos fijos por señal y `exp_c6` no reentrenó el recuperador.

## Evidencia bibliográfica usada

- Entrenar con la variedad de canal e interferencia que se encontrará (arXiv 1809.06176; OTA OFDM PMC13211014).
- Entrenamiento adversario de dominio (DANN) para cambio de canal Rayleigh/Rician (arXiv 2508.06829).
- Rasgos polares más robustos a desvanecimiento (arXiv 2001.01395) — ya existe la rama polar.
- La sincronía de símbolo coherente entre entrenamiento y uso es crítica (UMONS, Milcom).
- CMA solo iguala amplitud; necesita etapa de fase aparte (arXiv 2204.11776).
- DeepSets / PointNet para conjuntos sin orden (arXiv 1703.06114, 1612.00593).

## Unidades (una unidad = un archivo; contrato completo en el docstring de cada esqueleto)

| # | Archivo | Ejecutor | Contenido |
|---|---|---|---|
| U1 | `src/banco/canales_extra.py` | hydrogen | EPA/EVA/ETU, Nakagami-m, sombra lognormal, dos rayos, ecos largos, Watterson HF, interferencia modulada adyacente y cocanal, barrido, pulsos |
| U2 | `src/cascada/sintesis_variada.py` | neon | señales de entrenamiento del recuperador con todos los canales del banco + extra, sps extremos |
| U3 | `src/cascada/seguimiento.py` | neon | CMA secuencial y fase por bloques |
| U4 | `src/cascada/tramos.py` | hydrogen | análisis por tramos y votación |
| U5 | `src/cascada/descriptor_puntos.py` | neon | descriptor DeepSets/PointNet sobre símbolos |
| U6 | `src/cascada/descriptor_variantes.py` | hydrogen | CNN ancha y cabeza adversaria de dominio |

Integración (Claude): `src/banco/canales.py` (`CANALES_EXTRA`, `TODOS_CANALES`), `src/banco/senal.py`
(buscar en `TODOS_CANALES`), `src/banco/config.py` (`SPS_EXTREMOS`, parámetro `extremos`),
`src/cascada/cadena.py` (ruta `red_seguimiento`).

Experimentos (Claude):
- `experiments/exp_c7_recuperador.py`: afina el recuperador actual en GPU con `sintesis_variada`.
- `experiments/pool_variado.py`: vistas con todos los canales y sps extremos; guarda símbolos recuperados.
- `experiments/exp_c8_arreglos_cnn.py`: compara variantes `imagen`, `ancha`, `puntos`, `adversaria`,
  `multivista` en particiones dejando fuera generador / familia de canal.

## Orden

1. Lote A: U1, U2, U3 (independientes). Verificar. Integrar canales y sps.
2. Entrenar recuperador (exp_c7) en segundo plano mientras corre el lote B: U4, U5, U6.
3. Generar pool variado con el recuperador nuevo.
4. Cribado de variantes CNN (4 particiones, 4000 pasos); la ganadora en todas las particiones.
5. Documentar en `docs/CASCADA_ROBUSTEZ.md` y actualizar el HTML del esquema.

## Estado

Se actualiza al final de cada paso.

- **Lote A verificado.** U1 (`canales_extra`, hydrogen) y U2 (`sintesis_variada`, neon) pasan su criterio de aceptación y los 102 tests del banco. U3 (`seguimiento`, neon) agotó los 15 min de agy: su registro mostró que la referencia "solo bloque anterior" del contrato resbalaba en 16QAM. La implementación que dejó en disco (DP + segunda pasada contra referencia global) pasa el criterio en 50 semillas de 16QAM/QPSK/8PSK/BPSK con deriva lineal y senoidal; se adoptó y el contrato se reescribió para describirla. El CMA de seguimiento baja el error de módulo de 0,44 a 0,20 frente a pesos fijos.
- **Lote B verificado.** U4 (`tramos`, hydrogen), U5 (`descriptor_puntos`, neon) y U6 (`descriptor_variantes`, hydrogen) pasan al primer intento.
- **Integración.** `cadena.py` tiene la ruta `seguimiento` y la cadena `mixta_seguimiento`; `pipeline.py` carga descriptores de cualquier variante (`imagen`, `ancha`, `puntos`, `adversaria`, `multivista`) según los metadatos del paquete. `armar_banco.py` acepta canales nuevos y `--extremos`; `pool_multigenerador.py` acepta `--variado`.
- **Datos.** Pool variado `data/cascada_variado` (29 600 vistas, los 32 canales, 20 % de tasas extremas). Banco `data/banco/canales2` (3 888 archivos, 12 canales nuevos, 30 % de tasas extremas, 2,5 a 32 sps).
- Tests: 450 pasan; los 7 que fallan (monitor y niveles 0–3) fallan igual con los archivos originales porque les faltan resultados del repositorio original.
- **Recuperador reentrenado** (`exp_c7`, 70 min en la RTX, 50 652 pasos): pérdida de validación 0,168 → 0,1245 (−26 %). Mayores mejoras: impulsivo 0,20 → 0,05, sombra 0,48 → 0,19, mp_largo 0,55 → 0,41, tdl_rayleigh 0,58 → 0,47, tdl_rician 0,55 → 0,43, lte_eva 0,22 → 0,17. Siguen difíciles ecos_largos (0,29) y lte_etu (0,27). Modelo en `results/cascada/modelos_robustos/recuperador.pt`.
- `exp_c8`: corregido un bucle infinito cuando una partición deja menos de 128 geometrías (mismo resguardo que `exp_c6`). Prueba de humo completa con las 5 variantes; cribado lanzado.
- **Cribado** (4 000 pasos, 4 particiones): gana `adversaria` (AUROC medio 0,791, mejor en desvanecimiento no visto, 69,7 % frente a 70,0 % de `ancha`). `puntos` sola es la peor (67,3 %). Todas superan a la referencia por 6 a 12 puntos.
- **Etapa completa** (`adversaria`, 8 000 pasos, 12 particiones): 76,3 % de media frente a 68,1 % de la referencia; gana en las 12. Modelo final en `results/cascada/modelos_robustos/paquete_adversaria.joblib`.
- **exp_c9** (2 400 archivos del banco, SNR ≥ 10): paquete robusto 73,7 % frente a 65,2 % (canales nuevos 71,5 % frente a 57,4 %), AUROC 0,797 frente a 0,783, pero desconocidas detectadas 41 % frente a 54 %. El seguimiento empeora con el recuperador nuevo (70,1 %) y los tramos dan 74,1 % (no significativo): la cadena queda en `red`.
- **Interfaz**: motor «Canales variados» (`robusto`). Tests de interfaz: 22 pasan.
- **Documentación**: `docs/CASCADA_ROBUSTEZ.md` y `docs/esquema_clasificador.html` (artefacto republicado).
- **Recalibración** (`exp_c10`, pedida por el usuario): calibración con `catalogo` fuera de la muestra de exp_c9 + `validacion` (geometrías distintas). La calibración completa da puntaje abierto 0,425 pero AUROC 0,762; desplazar +0,1 el intercepto de la puerta del pool (criterio: ≤ 10 % de conocidas como desconocidas a SNR ≥ 10) da 0,423 con AUROC 0,797 y 46,7 % de desconocidas detectadas. Adoptada la desplazada; la interfaz apunta a `paquete_adversaria_desplazada.joblib`.
- **Pendiente**: la detección de desconocidas sigue por debajo del original (46,7 % frente a 54,1 %); multitrayecto estático largo (mp_largo 29 %, ecos_largos 36 %).
