# Robustez de la cascada a canales, interferencias y sobremuestreo

Trabajo del 2026-11-24 (plan: `docs/superpowers/plans/2026-11-24-robustez-canales.md`). Objetivo: que la cascada
siga funcionando con canales, interferencias y tasas de muestreo que no estaban en su entrenamiento, y elegir el
arreglo de CNN que mejor responda a datos de generadores que no vio.

## Resumen

| Paquete | Exactitud (2 400 archivos del banco, SNR ≥ 10 dB) | Banco base | Canales nuevos | AUROC desconocidas | Desconocidas detectadas |
|---|---|---|---|---|---|
| Original (`modelos/paquete_red.joblib`) | 65,2 % | 73,2 % | 57,4 % | 0,783 | **54,1 %** |
| **Robusto (`modelos_robustos/paquete_adversaria_desplazada.joblib`)** | **73,7 %** | **76,0 %** | **71,5 %** | **0,797** | 46,7 % |

- La exactitud sube 8,5 puntos en total y 14 puntos en los 12 canales nuevos.
- La CNN separa mejor las modulaciones desconocidas (AUROC más alto), pero el punto de operación detecta menos que el
  original. Con la puerta calibrada en el pool detectaba 41,1 %; con la recalibración (sección 6) detecta 46,7 %.
- En la interfaz es el motor «Canales variados» (`robusto`).

## Qué se hizo

### 1. Canales nuevos (32 en total)

`src/banco/canales_extra.py` añade 12 canales a los 20 del banco:

- Perfiles 3GPP LTE EPA, EVA y ETU.
- Desvanecimiento Nakagami-m, sombra lognormal y Watterson HF (dos rayos con Doppler gaussiano).
- Dos rayos y ecos largos.
- Interferencia modulada adyacente y cocanal, barrido de tono y pulsos.

Todos se probaron a 2,5, 8 y 32 muestras por símbolo: largo, valores finitos, potencia, determinismo y parámetros
serializables.

### 2. Sobremuestreo extremo

`armar_banco.py --extremos` y `pool_multigenerador.py --variado` sortean una fracción de tasas fuera del rango normal,
de 2,5 a 32 muestras por símbolo. Los datos nuevos usan las siguientes proporciones:

| Conjunto | Contenido | Tasas extremas |
|---|---|---|
| Pool `data/cascada_variado` | 29 600 vistas, los 32 canales | 20 % |
| Banco de prueba `data/banco/canales2` | 3 888 archivos, solo los canales nuevos | 30 % |

### 3. Recuperador reentrenado (`exp_c7`)

El recuperador original solo había visto multitrayecto estático. Se afinó 50 652 pasos (70 min en la RTX 3080) con
señales de `sintesis_variada`: 26 canales de entrenamiento y sps extremos.

La pérdida de validación bajó de 0,168 a 0,1245 (−26 %). Por canal:

| Canal | Pérdida antes | Pérdida después |
|---|---|---|
| impulsivo | 0,20 | 0,05 |
| sombra | 0,48 | 0,19 |
| mp_largo | 0,55 | 0,41 |
| tdl_rayleigh | 0,58 | 0,47 |
| tdl_rician | 0,55 | 0,43 |

### 4. Arreglos de CNN (`exp_c8`)

Se compararon cinco descriptores con particiones que dejan fuera un generador o una familia de canal completa:

| Arreglo | Qué es |
|---|---|
| `imagen` | La CNN actual sobre dos imágenes de 64×64 |
| `ancha` | La misma CNN con el doble de canales |
| `puntos` | Red de conjuntos tipo PointNet/DeepSets sobre 512 símbolos |
| `adversaria` | La CNN actual con cabeza de dominio e inversión de gradiente (DANN). La cabeza aprende a reconocer el generador o la familia de canal y la inversión empuja a la huella a olvidarlo |
| `multivista` | Fusión tardía de `imagen` y `puntos` |

**Cribado** (4 000 pasos, 4 particiones). La referencia es el paquete original:

| Arreglo | GNU Radio fuera | Analógico fuera | Desvanecimiento fuera | LTE fuera | Media | AUROC medio |
|---|---|---|---|---|---|---|
| Referencia | 68,7 % | 69,8 % | 45,4 % | 64,3 % | 62,1 % | — |
| imagen | 75,2 % | 77,5 % | 53,7 % | 70,5 % | 69,2 % | 0,766 |
| ancha | 74,6 % | 78,4 % | 55,9 % | 71,0 % | **70,0 %** | 0,765 |
| puntos | 71,2 % | 72,4 % | 54,2 % | 71,5 % | 67,3 % | 0,710 |
| **adversaria** | 74,7 % | **79,4 %** | **57,2 %** | 67,6 % | 69,7 % | **0,791** |
| multivista | **75,7 %** | 77,7 % | 53,1 % | **72,5 %** | 69,8 % | 0,784 |

**Elección: `adversaria`.**

- Tiene el mejor AUROC.
- Es la mejor en el caso más difícil, el desvanecimiento que no vio.
- Queda a 0,3 puntos de la mejor exactitud media.
- Usa un solo modelo; `multivista` necesita dos.
- La red de puntos sola rinde menos que las imágenes.
- Ensanchar la CNN no mejora la detección de desconocidas.

**Etapa completa** (8 000 pasos, las 12 particiones). `adversaria` supera a la referencia en todas:

| Partición dejada fuera | Adversaria | Referencia |
|---|---|---|
| Generador del repositorio | 80,3 % | 70,4 % |
| Generador procedural | 76,6 % | 69,7 % |
| Generador en frecuencia | 72,7 % | 68,7 % |
| Generador analógico | 75,6 % | 69,8 % |
| GNU Radio | 74,5 % | 68,7 % |
| Multitrayecto | 61,2 % | 58,1 % |
| Desvanecimiento | 53,1 % | 45,4 % |
| LTE | 70,5 % | 64,3 % |
| Desvanecimiento extra (Nakagami, sombra, Watterson) | **75,8 %** | 38,6 % |
| Sincronía | 94,1 % | 93,0 % |
| Hardware | 95,2 % | 94,1 % |
| Interferencia | 86,3 % | 76,8 % |
| **Media** | **76,3 %** | 68,1 % |

El AUROC medio de la etapa completa es 0,801. El modelo final se entrenó con todas las vistas y está en
`results/cascada/modelos_robustos/`.

### 5. Ecualizador con seguimiento y votación por tramos (`exp_c9`)

| Módulo | Qué hace |
|---|---|
| `src/cascada/seguimiento.py` | CMA símbolo a símbolo, más una corrección de fase por bloques con programación dinámica y una segunda pasada contra un histograma global |
| `src/cascada/tramos.py` | Analiza la señal en tramos de 16 384 muestras y vota con el promedio de log-probabilidades |

| Modo | Paquete original | Paquete robusto | Segundos por archivo |
|---|---|---|---|
| `red` | 65,2 % | 73,7 % | 0,65 |
| `mixta_seguimiento` | 65,8 % | 70,1 % | 0,74 |
| `red_tramos` | 65,4 % | 74,1 % | 0,33 |

**Decisión: la cadena sigue siendo `red`.**

- Con el paquete original, el seguimiento ganaba 0,6 puntos. Con el robusto pierde 3,6: el recuperador nuevo ya
  absorbe la mayor parte de la variación lenta, y el CMA secuencial vuelve a deformar la constelación.
- La votación por tramos gana 0,4 puntos, que está dentro del ruido (±0,9 puntos con 2 400 archivos).
- Ambos modos quedan disponibles: `cadena="mixta_seguimiento"` y `analizar_por_tramos`.

### 6. Recalibración de la puerta de desconocidas (`exp_c10`)

La puerta se había calibrado con el pool, que tiene señales mucho más variadas que el banco. Se recalibró con archivos
del banco que `exp_c9` no usa:

| Calibración | Archivos |
|---|---|
| Conocidas | 3 152 archivos `catalogo` fuera de la muestra de prueba |
| Desconocidas | 2 784 archivos del grupo `validacion` (24QAM, 6PSK, ANILLOS12, HEX19) |

Las desconocidas de calibración son geometrías distintas de las 8 de prueba. Se compararon dos formas:

| Calibración | Exactitud | Conocidas correctas | Desconocidas detectadas | Puntaje abierto | AUROC |
|---|---|---|---|---|---|
| Pool (antes) | 73,7 % | 38,2 % | 41,1 % | 0,397 | **0,797** |
| Completa con el banco | 73,7 % | **40,8 %** | 44,1 % | **0,425** | 0,762 |
| **Puerta del pool desplazada +0,1** | 73,7 % | 37,8 % | **46,7 %** | 0,423 | **0,797** |

- **Completa.** Rehace temperatura, puerta y umbral. Mejora el punto de operación pero baja el AUROC: la puerta aprende
  las 4 geometrías de calibración y ordena peor las 8 de prueba.
- **Desplazamiento (elegida).** Conserva la puerta del pool, así que el orden de P(desconocida) y el AUROC no cambian,
  y solo mueve su intercepto. El desplazamiento es el mayor que deja como mucho un 10 % de conocidas marcadas como
  «desconocida» en la calibración con SNR ≥ 10 dB.
  - El criterio se fijó antes de medir en la prueba.
  - Maximizar la media de conocidas correctas y desconocidas detectadas llevaba a un extremo (+1,8): 98 % de
    desconocidas detectadas, pero las conocidas indeterminadas pasaban a «desconocida», que es una afirmación falsa.
- Casi ninguna desconocida se confunde con una modulación inscrita (≤ 0,8 % en la calibración). Las que no se detectan
  quedan como indeterminadas.

Paquetes en `results/cascada/modelos_robustos/`:

| Archivo | Calibración |
|---|---|
| `paquete_adversaria_desplazada.joblib` | La de la interfaz |
| `paquete_adversaria_recal.joblib` | Completa con el banco |
| `paquete_adversaria_pool.joblib` | Original del pool (respaldo) |

## Por canal (paquete robusto, cadena `red`, banco)

| Mejora fuerte | Antes | Después |
|---|---|---|
| sombra | 15 % | 92,5 % |
| Nakagami | 51 % | 89,8 % |
| pulsos | 73,5 % | 91,2 % |
| gr_dinamico | 15,8 % | 42,1 % |
| LTE ETU | 47,7 % | 68,2 % |
| Watterson | 34,2 % | 50 % |
| tdl_rayleigh | 33,3 % | 50 % |
| plano_lento | 68 % | 84 % |

| Empeora | Antes | Después |
|---|---|---|
| mp_corto | 87 % | 73,9 % |
| mp_largo | 41,7 % | 29,2 % |
| ecos_largos | 45,5 % | 36,4 % |

Hay entre 20 y 60 archivos por canal, así que diferencias de menos de 10 puntos pueden ser ruido. El multitrayecto
estático largo sigue siendo el punto débil: la pérdida del recuperador en esos canales bajó poco (0,41 en mp_largo,
0,29 en ecos_largos).

## Pendiente

- **Puerta de desconocidas.** Tras la recalibración detecta 46,7 %, todavía por debajo del 54,1 % del paquete
  original. La puerta solo ve rasgos de la mezcla gaussiana. Una puerta que mire también la huella completa, o un fondo
  con más geometrías raras en el pool, son los siguientes pasos.
- **Multitrayecto largo.** Para mp_largo y ecos_largos haría falta un recuperador con más memoria (ventana más larga)
  o un ecualizador clásico más largo como ruta alternativa.

## Fuentes

- O'Shea, Roy, Clancy, *Over-the-Air Deep Learning Based Radio Signal Classification* (arXiv 1712.04578), y
  arXiv 1809.06176: entrenar con la variedad de canal que se encontrará.
- Ganin et al., *Domain-Adversarial Training of Neural Networks* (arXiv 1505.07818). Aplicación a cambio de canal
  Rayleigh/Rician en AMC: arXiv 2508.06829.
- Zaheer et al., *Deep Sets* (arXiv 1703.06114); Qi et al., *PointNet* (arXiv 1612.00593).
- Rasgos polares y robustez a desvanecimiento: arXiv 2001.01395.
- CMA y su ambigüedad de fase, que exige una etapa de fase aparte: arXiv 2204.11776.
- Perfiles 3GPP EPA/EVA/ETU: TS 36.104, anexo B. Modelo de Watterson para HF: ITU-R F.1487.

## Para repetir

```bash
.venv/Scripts/python.exe -m experiments.exp_c7_recuperador
C:/Users/David/radioconda/python.exe -m experiments.pool_multigenerador --variado
C:/Users/David/radioconda/python.exe -m experiments.armar_banco --salida data/banco/canales2 --canales lte_epa,lte_eva,lte_etu,nakagami,sombra,dos_rayos,ecos_largos,watterson,int_adyacente,int_cocanal,int_barrido,int_pulsos --extremos 0.3 --semilla 2026112403 --trabajos 6 --por-celda 3
.venv/Scripts/python.exe -m experiments.exp_c8_arreglos_cnn --etapa cribado
.venv/Scripts/python.exe -m experiments.exp_c8_arreglos_cnn --etapa completo --variante adversaria
.venv/Scripts/python.exe -m experiments.exp_c8_arreglos_cnn --etapa final --variante adversaria
.venv/Scripts/python.exe -m experiments.exp_c9_seguimiento --paquete results/cascada/modelos_robustos/paquete_adversaria.joblib
.venv/Scripts/python.exe -m experiments.exp_c10_recalibrar --modo desplazamiento
.venv/Scripts/python.exe -m experiments.exp_c9_seguimiento --paquete results/cascada/modelos_robustos/paquete_adversaria_desplazada.joblib
```

Resultados en `results/cascada/arreglos_cnn/` (cribado y etapa completa) y `results/cascada/seguimiento/` (modos de
cadena con cada paquete).
