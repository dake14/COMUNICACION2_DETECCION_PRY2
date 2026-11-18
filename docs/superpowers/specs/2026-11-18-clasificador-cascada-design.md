# Clasificador en cascada independiente del generador

**Fecha:** 2026-11-18 · **Estado:** propuesta, pendiente de aprobación · **Entrega del curso:** semana del 1 de diciembre de 2026

## 1. Intención

El Proyecto 2 terminará con un desafío de archivos I/Q cuya modulación y canal no se revelan. Esos archivos pueden venir de varios proveedores o generadores, y pueden contener modulaciones que nunca usamos. Queremos un sistema que:

- no dependa de los detalles de nuestro generador;
- decida con probabilidades, no mirando una constelación;
- aprenda patrones sin usar etiquetas de modulación;
- use pesos especializados: unos para recuperar la señal y otros para identificarla;
- diga "desconocida" cuando algo no se parezca a lo que conoce.

La guía premia demostrar que el clasificador aprendió características generalizables y no particularidades del entrenamiento. Este trabajo apunta a eso.

### Evidencia que lo motiva

Cifras guardadas en `results/nivel5_diagnostico/n5_1_diagnostic_report.json` y `results/paper_research/`. No se reprodujeron porque los modelos no están en el repositorio.

| Hallazgo | Medida |
|---|---|
| El SVM256 falla con QPSK de GNU Radio cuando el CFO es exactamente 0 | 4 de 16 ventanas; con 5 Hz, 14 de 16 |
| El multitrayecto no visto es la mayor caída de `MultiDomainAMCNet` | 67,6 % → 36,9 % |
| El CFO no visto casi no la afecta | 67,6 % → 64,0 % |
| 64-QAM se confunde con 16-QAM a SNR ≥ 10 dB | 85 de 120 casos; cada ventana tiene solo 32 símbolos |
| La red actual está atada a un formato | 6 clases, 8 muestras por símbolo, pulso RRC con β de 0,25 a 0,50 |

## 2. Objetivos

1. Clasificar por archivo las seis modulaciones del proyecto con generadores no usados al entrenar.
2. Entregar una probabilidad por modulación, más "desconocida", calibrada en datos independientes.
3. Rechazar modulaciones nunca vistas y describirlas: número aproximado de puntos y de anillos.
4. Agrupar entre sí los archivos desconocidos que parezcan la misma modulación.
5. Permitir añadir una modulación al catálogo sin reentrenar redes.
6. Recuperar los símbolos de la señal bajo multitrayecto, ruido y errores de sincronía.

### Fuera de alcance

- Captura en vivo, celular, sonido o radio. Lo inalámbrico se retomará después.
- Nombrar OFDM, FSK, GMSK o modulaciones analógicas. Deben salir como desconocidas o indeterminadas, y se comprobará.
- Varias señales simultáneas en un mismo archivo.
- Reentrenar `MultiDomainAMCNet` o reescribir el artículo.

### Lo que haría fallar el proyecto aunque el código funcione

- Ajustar algo mirando el conjunto de prueba final.
- Un recuperador que "invente" puntos y convierta una modulación rara en una conocida.
- Publicar una confianza que no esté calibrada.
- Romper el monitor actual o sus pruebas.
- Que la decisión dependa de inspeccionar visualmente la constelación. La figura solo explica el resultado.

## 3. Enfoques considerados

| Enfoque | Ventaja | Limitación | Decisión |
|---|---|---|---|
| **Especialistas en cascada** | Ataca multitrayecto, dependencia del generador y modulaciones nuevas | Es el de más trabajo y tiene más etapas que validar | **Elegido por el usuario** |
| Solo aprendizaje sin clases + PCA | Simple y rápido | No corrige el multitrayecto, que es el fallo mayor | Descartado |
| Mantener la red actual y añadir rechazo de desconocidas | El más seguro para la entrega | Sigue atado a 6 clases y a 8 muestras por símbolo | Descartado |

## 4. Arquitectura

```text
archivo I/Q
   │
   ▼
Detección, espectro y SNR ............ código actual del monitor
   │
   ▼
1. Normalizador (clásico) ............ tasa de símbolo, CFO grueso, 8 muestras/símbolo
   │
   ▼
2. Recuperador R (pesos propios) ..... limpia canal y ruido, estima el instante de símbolo
   │
   ▼
3. Autofoco (clásico) ................ elimina el giro residual, entrega símbolos
   │
   ▼
4. Descriptor D (pesos propios) ...... huella de la constelación, aprendida sin clases
   │
   ▼
5. Decisor probabilístico ............ PCA + mezclas gaussianas + evidencia por archivo
   │
   ▼
modulación | DESCONOCIDA | INDETERMINADA, con probabilidades y descripción
```

R y D son modelos separados, con pesos y entrenamiento propios. R se entrena primero y se congela. D se entrena sobre la salida real de R y del autofoco, para que aprenda con los artefactos que verá en uso.

### Convenciones

- Formato canónico: 8 muestras por símbolo, en bloques de 128 símbolos (1024 muestras) con solape del 50 %.
- Todo se calcula en ciclos por muestra o por símbolo. `fs_hz` solo se usa para informar en Hz, así que **la cascada no la necesita**.
- La SNR sigue la definición de `src.channels.add_awgn`: potencia media por muestra.
- Longitud mínima: 256 muestras en el formato original. Con pocos símbolos se espera `INDETERMINADA`, no una respuesta forzada.

### Etapa 1: normalizador

Sin pesos.

1. Quita el nivel de continua y normaliza la potencia.
2. Estima el CFO grueso con el centro de la densidad espectral suavizada.
3. Busca la tasa de símbolo en las líneas cíclicas de `|x|²` y de la señal diferencial, usando **todo el archivo**, entre 2 y 32 muestras por símbolo, no necesariamente enteras.
4. Conserva hasta 3 hipótesis y remuestrea cada una a 8 muestras por símbolo.

Las hipótesis pasan por las etapas 2 y 3, y gana la que produzca la constelación más nítida. Si ninguna alcanza un mínimo de nitidez, el estado es `tasa_no_identificada` y el resultado final es `INDETERMINADA`. Un parámetro opcional `--sps` permite indicar la tasa cuando se conoce.

### Etapa 2: recuperador R

- **Entrada:** I/Q de 2 × 1024 valores a 8 muestras por símbolo.
- **Salida principal:** forma de onda limpia de 2 × 1024. Es el tren de símbolos original con un pulso canónico (coseno alzado, β = 0,35), sin ruido ni canal, alineado al trayecto más fuerte.
- **Salida auxiliar:** seno y coseno de la fase de muestreo dentro del símbolo.
- **Lo que no corrige:** el giro lento por CFO residual se conserva a propósito y lo resuelve el autofoco.
- **Red:** convolucional 1D residual con dilataciones o U-Net, con `GroupNorm` como el resto del proyecto. Su campo receptivo debe cubrir al menos 3 símbolos de retardo más la cola del pulso.
- **Pérdida:** error cuadrático complejo invariante a una fase global, que se alinea en forma cerrada, más la pérdida de temporización.
- **Archivo completo:** los bloques se recomponen por solape y suma.

**Salvaguarda contra alucinaciones:** R se entrena con constelaciones procedurales aleatorias, no con las seis del proyecto, para que no aprenda un alfabeto. Una prueba específica verifica que una constelación retenida conserve sus puntos.

**Referencia clásica obligatoria:** filtro adaptado, temporización de Gardner y ecualizador ciego CMA. Si R no la supera en el hito 2, la cascada usa la cadena clásica.

### Etapa 3: autofoco

Sin pesos.

1. Toma los símbolos en los instantes que indica R.
2. Busca, sobre una rejilla fina, la corrección de frecuencia que minimiza la entropía del histograma 2D de la constelación, usando varios bloques del archivo.

Este criterio no presupone simetría, por lo que sirve para constelaciones arbitrarias, a diferencia de elevar a la M-ésima potencia. La fase global queda libre y se trata con invariancia en la etapa 4. La salida incluye los símbolos y una medida de nitidez.

### Etapa 4: descriptor D

- **Entrada:** imágenes de densidad de 64 × 64 de los símbolos de cada bloque, en coordenadas cartesianas y polares.
- **Red:** CNN 2D pequeña con `GroupNorm`.
- **Salida:** huella de 128 dimensiones normalizada, y cabezas auxiliares geométricas: orden aproximado (`log2 M`) y número de anillos.
- **Aprendizaje sin clases:** contrastivo. Cada geometría procedural aleatoria es su propia identidad. Un par positivo son dos transmisiones independientes de la misma geometría, con datos, pulso, canal, SNR y sincronía distintos. Los negativos son otras geometrías. No se usa ninguna etiqueta "QPSK", "16-QAM", etc.
- **Invariancias reforzadas:** rotación de fase y conjugación.

### Etapa 5: decisor probabilístico

1. **PCA con blanqueo** sobre huellas, conservando alrededor del 95 % de la varianza.
2. **Catálogo:** por cada modulación conocida se generan ejemplos y se ajusta una mezcla gaussiana, condicionada a un rango de SNR estimado.
3. **Modelo de fondo:** una mezcla amplia ajustada sobre geometrías procedurales. Representa "cualquier otra cosa" y da la hipótesis `DESCONOCIDA` sin un umbral arbitrario.
4. **Evidencia por archivo:** se acumulan las log-verosimilitudes de todos los bloques. Una temperatura ajustada en validación compensa que los bloques no son independientes.
5. **Calibración** de la probabilidad final en validación.
6. **Decisión:**
   - modulación del catálogo si domina con claridad;
   - `DESCONOCIDA` si domina el modelo de fondo;
   - `INDETERMINADA` si ninguna hipótesis alcanza la confianza mínima.
7. **Por lotes:** las huellas de los archivos desconocidos se agrupan con HDBSCAN, por ejemplo: "U1: X03, X07, X11".

Añadir una modulación solo requiere generar ejemplos y ajustar su mezcla. R y D no cambian.

## 5. Datos

### Generador procedural de entrenamiento

**Geometrías:** PSK de cualquier orden, QAM cuadrada y en cruz, PAM, anillos tipo APSK, redes hexagonales, puntos aleatorios con distancia mínima, y versiones deformadas. Orden de 2 a 128 puntos, con probabilidades de símbolo no siempre iguales.

| Variable | Rango de entrenamiento |
|---|---|
| Pulso | RRC y coseno alzado con β de 0,1 a 1; gaussiano; semiseno; Kaiser; longitudes variables |
| Canal | De 1 a 6 trayectos complejos con retardo de hasta 3 símbolos; desvanecimiento Rayleigh por bloque |
| Sincronía | Fase aleatoria, CFO residual, desfase de temporización, error de tasa de hasta ±0,5 % |
| Receptor | Desbalance I/Q leve, nivel de continua, cuantización de 8 bits o más |
| SNR | De −5 a 30 dB |

### Modulaciones

| Grupo | Modulaciones | Uso |
|---|---|---|
| Catálogo | BPSK, QPSK, 8PSK, 16-QAM, 64-QAM, 4-PAM | Se inscriben en el decisor |
| Desconocidas de validación | 6-PSK, 24-QAM, red hexagonal de 19 puntos, 12 puntos en 3 anillos | Ajustan umbrales y temperatura |
| Desconocidas de prueba final | 32-QAM en cruz, 128-QAM en cruz, 16-APSK, 32-APSK, 12-PSK, 8-QAM en estrella, 2 geometrías arbitrarias con semilla reservada | Solo evaluación final |

El generador rechaza cualquier geometría demasiado parecida a una desconocida de prueba, tras alinear fase y escala.

### Generadores de evaluación

| Generador | Papel |
|---|---|
| Procedural propio | Entrenamiento e inscripción |
| `src.signals` + `src.waveform` + `src.channels` | Validación |
| GNU Radio 3.10.12 (radioconda instalado en esta PC) | Prueba final |
| TorchSig 2.2.0, en un entorno aparte | Prueba final. Si no se instala o no permite controlar las condiciones, se sustituye por un generador independiente escrito sin compartir código |
| Conjunto ciego de un compañero (sección 12 de la guía) | Prueba adicional recomendada |

Cada archivo es una realización independiente, con semillas separadas por partición. Esto cumple la restricción de la sección 10 de la guía. Los canales de `shift_channels` y canales nuevos se reservan para prueba.

## 6. Salidas e integración

Código nuevo en `src/cascada/`, separado por etapa, y experimentos en `experiments/exp_c*.py`.

- `python -m src.monitor --motor cascada ...` añade este camino. **El motor actual sigue siendo el predeterminado**.
- `python -m src.cascada.lote <carpeta>` procesa un conjunto y produce la tabla del desafío: señal, detectada, SNR estimada, modulación estimada y confianza.
- Detección, espectro y SNR reutilizan el código del monitor. Si la detección da `False`, no se clasifica. Si es indeterminada por falta de referencia de ruido, la cascada se ejecuta y el aviso se conserva.
- El JSON mantiene las claves actuales y añade dentro de `classification`: `motor`, `probabilidades`, `descripcion`, `grupo` y el estado de cada etapa.

## 7. Validación

**Métricas por generador y SNR:** exactitud, F1 macro y matriz de confusión con desconocidas; AUROC y falsos aceptados al 95 % de detección de desconocidas; ECE; EVM antes y después de R; error de tasa de símbolo; tiempo por archivo.

**Comparación:** SVM reentrenado con los experimentos existentes, cadena clásica y `MultiDomainAMCNet` si se obtiene su checkpoint.

**Metas propuestas** para archivos de 512 símbolos o más, SNR ≥ 10 dB y generadores no vistos. Son objetivos, no promesas, y se informará también lo que no se alcance.

| Meta | Valor |
|---|---|
| Exactitud en las seis modulaciones | ≥ 85 % y superior a las referencias |
| Sensibilidad en 64-QAM | ≥ 70 % |
| AUROC de desconocidas | ≥ 0,90 |
| ECE de la confianza final | ≤ 0,10 |
| Inferencia por archivo en CPU | ≤ 2 s |

**Prueba de catálogo abierto:** tras la evaluación final, se inscribe 16-APSK sin reentrenar y se mide si pasa de desconocida a reconocida, en datos nuevos.

## 8. Hitos y puntos de decisión

| Hito | Fecha objetivo | Contenido | Se sigue si |
|---|---|---|---|
| H0 Viabilidad | 20 nov | Generador, normalizador y autofoco, sin redes | Tasa con error < 1 % en ≥ 90 % de archivos a SNR ≥ 5 dB, y constelación recuperable con receptor clásico |
| H1 Base | 21 nov | RTX activa, datos y referencias reproducibles | Entrenamiento corre en GPU |
| H2 Recuperador | 23 nov | R contra la cadena clásica | Mejora de EVM ≥ 3 dB con multitrayecto no visto a SNR ≥ 10 dB. Si no, se usa la cadena clásica |
| H3 Identificación | 26 nov | D y decisor en validación | Supera a las referencias en validación |
| H4 Integración | 27 nov | Monitor, lote, pruebas | Las pruebas nuevas pasan y las actuales no empeoran |
| H5 Evaluación final | 28 nov | Una sola ejecución sobre prueba | Resultados guardados con hashes |
| H6 Cierre | 29 nov | Informe y diapositivas | Listo para presentar |

**Si falta tiempo, se recorta en este orden:** agrupación por lotes, TorchSig, cabezas descriptivas, R (queda la cadena clásica). No se recortan la separación de datos, la calibración ni la evaluación con un generador no visto.

## 9. Riesgos

| Riesgo | Mitigación |
|---|---|
| Con β bajo y archivos cortos, la tasa de símbolo es difícil de estimar | Usar todo el archivo, varias hipótesis y `INDETERMINADA` como salida válida |
| R borra la estructura de una modulación nueva | Constelaciones aleatorias y prueba específica |
| Los archivos del profesor son demasiado cortos para 64-QAM | Informar la incertidumbre, sin forzar una respuesta |
| La GPU no está disponible | Hoy PyTorch no ve la RTX. Hay que activarla en el modo GPU de Armoury Crate antes de H1 |
| Plazo de unos 12 días | Puntos de decisión con alternativas clásicas |
| Faltan los modelos de Luis | La cascada no los necesita. Solo limitan la comparación con `MultiDomainAMCNet` y las 7 pruebas que hoy fallan en este clon |

## 10. Repositorio

- Se trabaja en una rama local `cascada`. La cuenta actual no tiene permiso de escritura en `LURAYD/COMUNICACION2_DETECCION_PRY2`, así que compartirlo requerirá un fork o que Luis lo incorpore.
- `.gitignore` excluye `*.pt` y `*.joblib`. Para que los modelos sean un entregable real, se publicarán como adjuntos de una release o por otra vía acordada.
- Entorno probado aquí: Python 3.12.10 y PyTorch 2.11 con CUDA 12.8. El proyecto original usó Python 3.11 y PyTorch 2.10.

## 11. Queda a criterio de quien lo construya

Tamaños exactos de las redes, optimizador, volumen de datos por época, rejilla del autofoco, número de componentes de las mezclas y organización interna de los módulos, siempre que respeten las interfaces, la separación de datos y las metas de esta especificación.
