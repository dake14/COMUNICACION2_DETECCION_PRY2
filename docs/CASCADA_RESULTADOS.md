# Clasificador en cascada: resultados

Evaluación del 2026-11-18, rama `cascada`, solo CPU (Ryzen 9 5900HS).
Paquete evaluado: `results/cascada/modelos/paquete.joblib`, SHA-256 `5a8861f7bda02158…`.
Métricas completas: `results/cascada/final/metricas.json`.

## Resumen

La cascada identifica la modulación sin conocer la tasa de símbolo ni el generador, y da una probabilidad para cada
opción. En dos generadores que no se usaron para entrenar ni ajustar supera a la SVM de referencia. Cumple todas las
metas con GNU Radio. Con el generador independiente no alcanza dos: exactitud (79,2 % frente a 85 %) y AUROC de
desconocidas (0,839 frente a 0,90).

El sistema prefiere no responder antes que equivocarse. Cuando da una modulación del catálogo acierta el 93 % o más,
pero deja sin decidir alrededor de un tercio de las señales conocidas.

## Resultados finales

SNR ≥ 10 dB, archivos de 512 símbolos o más. Cada generador aporta 192 archivos del catálogo y 256 de modulaciones
desconocidas. Los intervalos son de Wilson al 95 %.

| Métrica | Meta | GNU Radio 3.10 | Independiente |
|---|---|---|---|
| Exactitud en las seis modulaciones | ≥ 85 % | **85,9 %** (80,3–90,2) | **79,2 %** (72,9–84,3), no cumple |
| Sensibilidad en 64QAM | ≥ 70 % | **96,9 %** (84,3–99,4) | **90,6 %** (75,8–96,8) |
| AUROC de desconocidas | ≥ 0,90 | **0,910** | **0,839**, no cumple |
| ECE de la confianza | ≤ 0,10 | **0,068** | **0,052** |
| Tiempo por archivo, percentil 95 | ≤ 2 s | **1,71 s** | **0,63 s** |
| Tasa de símbolo correcta (todas las SNR) | | 99,8 % | 94,8 % |

La exactitud de la tabla obliga a elegir una de las seis modulaciones. La salida real también puede ser `DESCONOCIDA`
o `INDETERMINADA`:

| Salida con SNR ≥ 10 dB | GNU Radio | Independiente |
|---|---|---|
| Conocidas con la modulación correcta | 57,8 % | 52,1 % |
| Conocidas marcadas `INDETERMINADA` | 35,4 % | 33,3 % |
| Conocidas marcadas `DESCONOCIDA` | 2,6 % | 13,5 % |
| Acierto cuando afirma una modulación del catálogo | 93,3 % | 98,0 % |
| Desconocidas marcadas `DESCONOCIDA` | 54,3 % | 51,6 % |
| Desconocidas marcadas `INDETERMINADA` | 41,4 % | 46,5 % |
| Desconocidas aceptadas como del catálogo | 4,3 % | 2,0 % |

### Comparación con la SVM

La SVM se entrenó solo a 8 muestras por símbolo, así que la comparación justa es a esa tasa. Hay pocos archivos en ese
subconjunto (36 y 42), por lo que las cifras son orientativas.

| SNR ≥ 10 dB | Cascada, todas las tasas | SVM, todas las tasas | Cascada, 8 sps | SVM, 8 sps |
|---|---|---|---|---|
| GNU Radio | 85,9 % | 62,5 % | 94,4 % | 69,4 % |
| Independiente | 79,2 % | 51,6 % | 73,8 % | 57,1 % |

`MultiDomainAMCNet` no se pudo comparar porque el clon del repositorio no trae sus pesos. TorchSig no se usó.

### Puntos débiles

- BPSK con GNU Radio: 59,4 %. Parte de los errores van a 8PSK. En desarrollo, esa confusión apareció con multitrayecto
  de retardo corto.
- 16QAM con el generador independiente: 59,4 %.
- A 10 sps con GNU Radio la exactitud baja a 69,4 %.
- Cerca de la mitad de las modulaciones desconocidas queda como `INDETERMINADA`, sobre todo las parecidas a las del
  catálogo: 12PSK, 128QAM y 8QAM en estrella.
- La confianza está peor calibrada a 10 dB con GNU Radio (ECE 0,185).

### Inscribir una modulación nueva

Se inscribió 16APSK sin reentrenar ninguna red. Antes se reconocía en 0 de 24 archivos y después en 24 de 24. Estos 24
archivos usan el generador del repositorio, no los generadores de la prueba final.

## Cómo funciona

1. **Normalizador clásico.** Estima tasa de símbolo y CFO grueso con líneas cíclicas, y remuestrea a 8 sps.
2. **Recuperación.** Compara tres rutas: red dilatada, filtro adaptado con CMA fraccionado y filtro adaptado solo. Cada
   ruta se refina con los modos de su propia constelación, y se conserva la de menor entropía.
3. **Autofoco.** Elimina el giro residual minimizando la entropía.
4. **Descriptor contrastivo.** Una CNN convierte la densidad de la constelación en una huella de 128 valores. Se entrena
   con geometrías aleatorias, sin etiquetas de modulación.
5. **Decisor.** PCA, una mezcla gaussiana por modulación y otra de fondo. Una regresión logística estima la probabilidad
   de que la señal sea desconocida usando rasgos que no dependen del número de clases, por lo que inscribir una
   modulación no exige recalibrar.

## Decisiones y desviaciones del diseño

- **La red de recuperación no pasó su criterio.** Mejora 0,68 dB sobre la cadena clásica en canales no vistos y la cadena
  mixta 2,84 dB; la regla pedía 3 dB. El diseño indicaba usar la cadena clásica. Se entrenaron y validaron ambas, y la
  mixta se eligió por clasificar mejor en validación (puntaje 0,693 frente a 0,613, con ventaja mínima exigida de 0,01).
  Es una desviación deliberada de la regla.
- **Tasa de símbolo.** El diseño elegía la hipótesis más nítida, pero esa regla acertaba solo el 82 % de las veces y el
  40 % en 64QAM. Se reemplazó por la línea cíclica más fuerte.
- **Calibración.** La temperatura con previa fija daba AUROC 0,82 en desarrollo. Se sustituyó por la regresión logística
  descrita arriba. El ajuste usó datos del generador del repositorio y del generador procedural.
- **Sin GPU.** La RTX estaba apagada. Por eso se usó Oerder-Meyr en lugar de una cabeza de temporización y redes de
  menos de 300 000 parámetros.

## Validez de la prueba final

- Los modelos se fijaron en un commit antes de la primera ejecución, y los conjuntos finales no se usaron para ajustar
  nada.
- La primera ejecución dio 0 % de tasa correcta en GNU Radio y la SVM quedó al azar. La causa era un error del generador:
  el filtro RRC recibía `1/sps` como tasa de símbolo y estrechaba la señal. Se corrigió el generador, se regeneraron sus
  archivos con la misma semilla y se repitió la evaluación con el mismo paquete.
- Los resultados inválidos están en `results/cascada/final_v1_gnuradio_invalido/`. Los del generador independiente
  fueron idénticos en ambas ejecuciones.
- Tras ver estos resultados no se cambió ningún modelo. Cualquier mejora futura necesitará conjuntos de prueba nuevos.
- Todas las señales son sintéticas. No se probó con capturas de radio reales.

## Reproducir

Desde la raíz del proyecto:

```bash
.venv/Scripts/python.exe -m experiments.exp_c0_viabilidad
.venv/Scripts/python.exe -m experiments.exp_c1_recuperador
.venv/Scripts/python.exe -m experiments.exp_c1b_comparar
.venv/Scripts/python.exe -m experiments.exp_c2_descriptor --cadena mixta --minutos 45
.venv/Scripts/python.exe -m experiments.exp_c3_inscripcion --cadena clasica
.venv/Scripts/python.exe -m experiments.exp_c3_inscripcion --cadena mixta
.venv/Scripts/python.exe -m experiments.exp_c3_inscripcion --elegir
C:/Users/David/radioconda/python.exe -m experiments.gen_gnuradio_cascada
.venv/Scripts/python.exe -m experiments.gen_independiente
.venv/Scripts/python.exe -m experiments.exp_c4_final
```

`exp_c4_final` se niega a correr si ya existe `results/cascada/final/metricas.json`.

## Uso

Tabla del desafío para una carpeta de archivos I/Q:

```bash
.venv/Scripts/python.exe -m src.cascada.lote carpeta_de_senales --salida results/cascada/lote
```

Un archivo con el monitor:

```bash
.venv/Scripts/python.exe -m src.monitor --input senal.c64 --motor cascada --output salida.json --plot salida.png
```

Si se conoce la tasa, ambos comandos aceptan `--sps`.
