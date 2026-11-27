# Comparación de arreglos de la cascada

Evaluación del 2026-11-18. Datos completos en `results/cascada/arreglos/tabla.json`. Para repetirla:

```bash
C:/Users/David/radioconda/python.exe -m experiments.gen_arreglos gnuradio
.venv/Scripts/python.exe -m experiments.gen_arreglos independiente
.venv/Scripts/python.exe -m experiments.exp_c5_arreglos
```

## Método

- Señales nuevas, con semillas no usadas antes: GNU Radio y generador independiente.
- Las cifras usan SNR ≥ 10 dB: 768 archivos de modulaciones conocidas y 1024 de desconocidas.
- Cada arreglo inscribe el catálogo y calibra su propio decisor con las semillas de H3.
- Todos comparten la CNN descriptora, entrenada con la cadena `mixta`. Esto puede favorecer a los arreglos parecidos.
- El tiempo es la mediana por archivo sobre 80 archivos, uno a la vez, con un hilo de CPU.
- Regla fijada antes de ver resultados: gana el arreglo más rápido que quede a 2 puntos o menos del mejor en exactitud y
  en puntaje abierto. El puntaje abierto promedia conocidas correctas en la salida real y desconocidas detectadas.

## Resultados

| Arreglo | Exactitud | GNU Radio | Independiente | Conocidas correctas | Desconocidas detectadas | Falsa aceptación | AUROC | CPU (ms) | RTX (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **Red + refino (el más eficiente)** | **81,0 %** | 86,7 % | 75,3 % | 54,6 % | 52,5 % | 3,4 % | 0,872 | **68** | **49** |
| Mixta sin refino | 79,8 % | 82,0 % | 77,6 % | 50,8 % | 57,6 % | 2,6 % | 0,875 | 107 | 83 |
| Red sin refino | 79,4 % | 82,8 % | 76,0 % | 51,6 % | 51,5 % | 4,2 % | 0,869 | 59 | 40 |
| Mixta + refino (actual) | 79,0 % | 83,1 % | 75,0 % | 53,8 % | 49,3 % | 3,8 % | 0,848 | 172 | 145 |
| CMA y directa + refino | 76,4 % | 81,5 % | 71,4 % | 48,0 % | 58,3 % | 1,2 % | 0,872 | 109 | 122 |
| CMA sin refino | 74,1 % | 77,3 % | 70,8 % | 44,7 % | 48,5 % | 1,3 % | 0,871 | 38 | 40 |
| CMA + refino | 74,0 % | 76,3 % | 71,6 % | 44,4 % | 49,3 % | 1,0 % | 0,890 | 42 | 51 |
| Directa + refino | 71,0 % | 71,9 % | 70,1 % | 35,3 % | 34,7 % | 2,2 % | 0,789 | 35 | 38 |
| Directa sin refino | 67,2 % | 64,8 % | 69,5 % | 33,3 % | 35,8 % | 2,4 % | 0,794 | 43 | 31 |
| SVM de referencia | 54,6 % | 59,4 % | 49,7 % | no aplica | no aplica | no aplica | no aplica | 17 | no aplica |

La exactitud obliga a elegir una de las seis modulaciones. Su intervalo de confianza al 95 % es de unos ±3 puntos.

## Lectura

- `red + refino` iguala al arreglo actual dentro del margen estadístico: acierta 2 puntos más, pero la diferencia no es
  significativa (McNemar, p = 0,105). Sus ventajas claras son el tiempo, 2,5 veces menor, y un AUROC algo mayor.
- Los arreglos sin la red pierden exactitud. Frente al actual, CMA pierde 4,9 puntos (p = 0,0004) y directa + refino 8
  puntos (p < 0,00001).
- La RTX ayuda solo a los arreglos con red. En los clásicos no aporta o empeora el tiempo.
- Ningún arreglo llega a 85 % de exactitud conjunta ni a AUROC 0,90 en estos datos.
- Elegir el mejor de nueve con estos mismos datos hace su cifra algo optimista.

## Trabajos relacionados

Las cifras son de cada autor, con sus propios datos, y no se comparan directamente con las de esta tabla.

| Tema | Trabajo | Qué aporta |
|---|---|---|
| Ecualizar y clasificar juntos | [EMC2-Net](https://arxiv.org/abs/2303.10934), ICASSP 2023 | Trabaja sobre puntos de constelación, no imágenes, con menor complejidad |
| Procesamiento de señal y redes | [Red de doble camino](https://arxiv.org/abs/2106.10543) | Cerca de 5 % más en clasificación y hasta 15 % en señales largas |
| Procesamiento de señal y redes | [Red con procesamiento lineal](https://arxiv.org/abs/2006.00729) | La red estima parámetros y el procesamiento lineal corrige CFO y multitrayecto |
| Ecualización ciega | [CMA y prueba de Kuiper](https://arxiv.org/abs/1903.07657) | CMA antes de clasificar con canal selectivo desconocido |
| Cambio de canal | [CLSR-AMC](https://par.nsf.gov/biblio/10428343-contrastive-learning-self-reconstruction-channel-resilient-modulation-classification), INFOCOM 2023 | Contraste más reconstrucción: entre 15,97 y 24,29 % de ganancia media frente a su versión contrastiva |
| Modulaciones desconocidas | [MMPRF](https://doi.org/10.1109/TPAMI.2023.3294505), TPAMI 2023 | Prototipos con margen, rasgos de señal y muestras desconocidas generadas |
| Modulaciones desconocidas | [CIR](https://arxiv.org/abs/2312.13023) | Usa el error de reconstrucción para separar desconocidas |
| Pocos ejemplos | [FSOS-AMC](https://arxiv.org/abs/2410.10265), IEEE IoT Journal 2025 | Conjunto abierto con pocos ejemplos y multitrayecto |
| Sin etiquetas | [RIS-MAE](https://arxiv.org/abs/2508.00274) | Autoencoder enmascarado sobre I/Q crudo |
| De sintético a real | [Embeddings para señales RF](https://arxiv.org/abs/2510.23186) | Entrena con protocolos sintéticos y valida con señales reales |
| Constelaciones | [CNN de mapa de constelación](https://arxiv.org/abs/2009.02026), GLOBECOM 2020 | Cerca de 87 % a 0 dB con Rayleigh multitrayecto, ocho modulaciones |
| Modelos livianos | [LENet](https://doi.org/10.3390/electronics12081820) | Familia de redes; la menor tiene unos 4,61 mil parámetros |
| Modelos livianos | [DP-DRSN](https://arxiv.org/abs/2507.04586) | Unos 27 mil parámetros y 62,13 % medio en RML2018.01a |
| Datos variados | [Sig53 y TorchSig](https://arxiv.org/abs/2207.09918) | 53 clases sintéticas con imperfecciones y generación durante el entrenamiento |
