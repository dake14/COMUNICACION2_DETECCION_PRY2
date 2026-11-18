# COM2 - Proyecto 2

## Descripción del Proyecto
Proyecto desarrollado para la asignatura de Comunicaciones 2 (COM2).

Las fases 1–5 cubren seis moduladores, conformación RRC, canales AWGN/fase/CFO/multipath, datasets reproducibles, detección, espectro, estimación de SNR con referencia de ruido y clasificadores convencionales. E1 mide detección y falsas alarmas. E2 compara SVM y bosque aleatorio en CPU para 256 y 512 muestras. [Fase 1](docs/FASE1_VALIDACION.md) · [Fase 2](docs/FASE2_VALIDACION.md) · [Fase 3](docs/FASE3_VALIDACION.md) · [Fase 4 y E1](docs/FASE4_VALIDACION.md) · [Fase 5 y E2](docs/FASE5_VALIDACION.md)

## Ejecutar desde la raíz del proyecto

Entorno comprobado: Python 3.11.0. Para preparar uno nuevo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-fase5.txt
```

Pruebas y figuras reproducibles:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m experiments.diagnostico_fase1
.\.venv\Scripts\python.exe -m experiments.diagnostico_fase2
```

Los diagnósticos generan PNG y métricas JSON en `results/fase1/` y `results/fase2/`, sin ventanas interactivas. Para abrir la demostración manual BPSK:

```powershell
.\.venv\Scripts\python.exe -m src.test_signal
```

Se usan imports `from src.signals import qpsk_modulation` y `from src.waveform import pulse_shape`. Las funciones RRC también siguen disponibles en `src.signals` por compatibilidad. Ejecuta módulos con `python -m`; la ejecución directa `python src/test_signal.py` ya no es el contrato documentado.

Consulta [las tablas de bits y símbolos](docs/MAPPING_MODULACIONES.md) y [el modelo de señal](docs/MODELO_DE_SENAL.md) para interpretar energía, muestras y retardo. Los originales de Fase 0 se conservan en `results/fase1/antes/`; el proyecto cuenta con repositorio Git local (commit base `7e81d0df`) y manifiesto SHA-256 en [Nivel 0 Evidencia](docs/NIVEL0_EVIDENCIA_Y_PROTOCOLO.md).

Los [canales y sus unidades](docs/CANALES.md) se importan desde `src.channels`: `add_awgn`, `apply_carrier_impairments` y `apply_multipath`. El canal se aplica a la trama completa antes de segmentar; `sample_offset` permite mantener continuidad de fase/CFO si se procesan porciones. La copia de `channels.py` previa a Fase 2 está en `results/fase2/antes/`.

## Dataset de humo

Ya generado en `data/fase3_smoke/`: 224 realizaciones madre y 448 ventanas 256/512, con seis clases balanceadas y ruido solo. Configuración en [configs/dataset_smoke.json](configs/dataset_smoke.json); [protocolo](docs/PROTOCOLO_EXPERIMENTAL.md) y evidencia resumida en `results/fase3/`.

Para regenerar en una carpeta nueva:

```powershell
.\.venv\Scripts\python.exe -m src.dataset --config configs/dataset_smoke.json --output data/fase3_smoke_nueva
```

El generador rechaza destinos existentes. Las mismas configuración y semilla reproducen el contenido. `data/` se ignora en Git por ser regenerable; configuración, código y resultados resumidos se conservan.

`from src.dataset import load_observation` permite leer un registro usando su ID administrativo de `index.csv`. Devuelve solo I/Q y fs; etiquetas y parámetros verdaderos están en `oracle/manifest.jsonl`, accesible por separado mediante `load_truth`. La ventana de 256 es el prefijo de la de 512, no otra realización. `test_shift` queda reservado para evaluar cambio de canal. Este smoke no es suficiente para entrenar o medir precisión final.

## Detección, espectro y SNR

El experimento E1 puede repetirse en una carpeta nueva desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp01_detection --config configs/exp01_detection.json --output results/e1_repeticion
```

La ejecución registrada se encuentra en [results/fase4](results/fase4/). Para usar los tres bloques sobre una captura, el receptor necesita ventanas de ruido de la misma banda y ganancia. Este ejemplo supone que `noise_windows` contiene esas ventanas de 256 muestras y que `iq` es una captura independiente de la misma longitud.

```python
from src.detection import calibrate_energy
from src.monitor import analyze
from src.snr_estimation import calibrate_noise_reference

detector = calibrate_energy(noise_windows, pfa=0.01)
reference = calibrate_noise_reference(noise_windows.ravel())
result = analyze(iq, fs_hz=8000, detection_calibration=detector, noise_reference=reference)
```

Usa `fs_hz=None` si desconoces la frecuencia de muestreo. El eje espectral se expresará en ciclos/muestra. Si falta la referencia de ruido, una detección entregará SNR indeterminada. [Supuestos y unidades](docs/FUNDAMENTOS_FASE4.md)

## Clasificación convencional

E2 seleccionó dos SVM RBF con validación independiente. En test, exactitud de 43,69 % y 46,60 % para 256 y 512 muestras, respectivamente. El conjunto cubre ocho niveles de SNR de −20 a 20 dB. La inferencia presupone una señal presente de una de las seis clases. [Resultados, límites y tiempos](docs/FASE5_VALIDACION.md)

```python
import numpy as np
from src.conventional import load_model, predict_observation

model = load_model("results/fase5/conventional_256.joblib")
with np.load("results/fase5/demo_new_observations.npz", allow_pickle=False) as data:
    iq = data["iq512"][0, :256]
result = predict_observation(model, iq=iq, fs_hz=8000)
print(result["modulation"])
```

Para repetir E2 en una carpeta nueva:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp02_conventional --config configs/exp02_conventional.json --output results/e2_repeticion
```

El dataset `data/e2_dataset` conserva 19 456 padres y prefijos exactos. Los mismos padres permiten la comparación posterior con el modelo avanzado. La configuración y las dependencias están en [configs/exp02_conventional.json](configs/exp02_conventional.json) y [requirements-fase5.txt](requirements-fase5.txt).

## Clasificación Avanzada (`SmallCNN` Fase 6 y `MultiDomainAMCNet` Exp 12)

Se evaluaron dos arquitecturas neuronales en GPU (RTX 4060) sobre conjuntos separados (que no deben mezclarse en una única comparación):
- **`SmallCNN` (Fase 6, 94 726 parámetros, `data/e2_dataset` y `data/fase6_holdout_sellado`):** En E3 (`e2_dataset/test`) obtuvo 40,94% y 18,60% bajo *shift*; en el holdout (`data/fase6_holdout_sellado`, semilla `2026110609`) obtuvo 40,75% en `test` y 19,02% en `test_shift` (frente a 43,27% / 26,81% de SVM256 y 47,21% / 29,06% de SVM512).
- **`MultiDomainAMCNet` (Exp 12 / artículo, `data/adaptive_dataset`):** Entrenada con síntesis física en GPU; en `adaptive_dataset/test` ($n=2400$) obtuvo 51,5833% nominal y 30,0833% en `test_shift` (frente a 43,75% / 30,33% de SVM256 y 46,04% / 29,54% de SVM512).
- **Diagnóstico de canal:** Aislamiento del impacto de CFO vs multitrayecto en `docs/DIAGNOSTICO_CANAL_RESULTADOS.md`. [Reporte Fase 6](docs/FASE6_VALIDACION.md) · [Holdout](docs/FASE6_HOLDOUT_VALIDACION.md)

## Monitor Integrado CLI para Archivos Desconocidos (Fase 7 · Corregido Nivel 1)

Herramienta para evaluar una señal desconocida generando las cinco salidas del proyecto bajo contrato explícito:

```powershell
.\.venv\Scripts\python.exe -m src.monitor --input ruta/a/senal.c64 --noise-ref ruta/a/ruido.c64 --fs-hz 8000
```

Soporta formatos `.c64`, `.bin`, `.npy`, `.npz` y `.csv` (rechaza lotes 2D, binarios truncados o valores no finitos con código `2`). Sin `--noise-ref` ni `--threshold` manual reporta detección indeterminada (`indeterminate_uncalibrated`); sin `--fs-hz` expresa el espectro en `ciclos/muestra` y marca la clasificación como `OUT_OF_SCOPE_MISSING_FS`. Genera JSON y figura PNG de PSD/constelación. [Documentación Monitor](docs/FASE7_MONITOR.md) · [Validación Nivel 2](results/nivel2_validacion/)

## Interoperabilidad con GNU Radio Companion (Fase 8)

Flowgraphs reproducibles en GNU Radio 3.10.9.2 sin necesidad de hardware RF:
- `grc/flowgraph_tx_channel.grc`: Demostración interactiva en Qt GUI con controles deslizantes en tiempo real para SNR (dB), CFO (Hz) y multitrayecto, junto a visualizadores de constelación, PSD y osciloscopio.
- `grc/flowgraph_headless_sim.grc`: Simulación desasistida por lotes ($N=4096$ muestras complex64).
- `grc/flowgraph_rx_from_file.grc`: Visualizador en GRC para señales generadas en Python.

Para abrir la interfaz gráfica:
```cmd
C:\Users\Luisr\radioconda\Scripts\gnuradio-companion.exe grc/flowgraph_tx_channel.grc
```

Para ejecutar la prueba de interoperabilidad automática:
```powershell
.\.venv\Scripts\python.exe -m experiments.exp11_gnuradio_demo
```
[Reporte Fase 8](docs/FASE8_GNURADIO.md)

## Investigación Adaptativa (Exp 10 y Exp 12)

Evaluación experimental de la hipótesis de selección dinámica de acciones (`SVM256`, `SVM512`, red neuronal) guiada por riesgo y latencia. En `results/paper_research/ablation_and_policy_metrics.json` (`data/adaptive_dataset/test`), Fast LBAP alcanzó 51,625% nominal frente a 51,5833% de `MultiDomainAMCNet` sola; la validación completa de paquete desplegable, latencia integral y contrastes pareados corresponde al Nivel 3 del plan de trabajo ([Plan y Criterios](docs/PLAN_MODIFICACIONES_Y_CRITERIOS_DE_ACEPTACION.md)). [Protocolo Adaptativo](docs/ADAPTATIVO_PROTOCOLO.md)

## Clasificador en cascada (independiente del generador)

Motor opcional que no necesita conocer la tasa de símbolo, entrega probabilidades y puede responder `DESCONOCIDA` o `INDETERMINADA`. Admite modulaciones nuevas sin reentrenar las redes. En generadores no usados al entrenar, a SNR ≥ 10 dB, obtuvo 85,9 % de exactitud con GNU Radio y 79,2 % con un generador independiente; este último no alcanza la meta de 85 %. [Resultados, limitaciones y reproducción](docs/CASCADA_RESULTADOS.md)

```powershell
python -m src.cascada.lote carpeta_de_senales --salida results/cascada/lote
python -m src.monitor --input senal.c64 --motor cascada --output salida.json --plot salida.png
```

## Estructura del Directorio
```text
COM2_Proyecto2/
├── .venv/            # Entorno virtual de Python
├── configs/          # Configuraciones JSON reproducibles
├── data/             # Datasets sintéticos e independientes
├── docs/             # Reportes técnicos de validación por fases
├── experiments/      # Scripts ejecutables de experimentos (E1–E6, Exp 10, Exp 11)
├── grc/              # Flowgraphs de GNU Radio Companion (.grc y .py)
├── results/          # Métricas JSON, matrices de confusión y gráficos PNG
├── src/              # Código fuente modular (signals, waveform, channels, monitor, etc.)
├── tests/            # Suite completa de pruebas unitarias e integración (pytest)
└── README.md         # Documentación general
```

## Configuración del Entorno Virtual

### 1. Activar el entorno virtual (Windows)
En PowerShell:
```powershell
.\COM2_Proyecto2\.venv\Scripts\Activate.ps1
```
O si estás dentro de la carpeta `COM2_Proyecto2`:
```powershell
.\.venv\Scripts\Activate.ps1
```

En CMD:
```cmd
COM2_Proyecto2\.venv\Scripts\activate.bat
```

### 2. Desactivar el entorno
```bash
deactivate
```
