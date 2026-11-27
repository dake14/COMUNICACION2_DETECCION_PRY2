# Modulaciones digitales básicas: etapa 1, generadores

## Intención

El usuario quiere que el clasificador reconozca **todas las modulaciones digitales básicas**, no solo las 6 lineales
actuales. El plan completo va por etapas:

1. Generadores (este documento).
2. Enrutador de familias con especialistas (espectro, potencias x², x⁴ y x⁸, envolvente, frecuencia instantánea) y
   fusión por regresión logística: P(modulación) = P(familia) × P(modulación | familia).
3. Equivalencias y lista conforme («también puede ser…»).
4. Especialistas dentro de cada familia.

## Clases (18)

| Familia | Clases |
|---|---|
| Amplitud | OOK, 4ASK (unipolar), 4PAM |
| Fase | BPSK, QPSK, 8PSK, OQPSK, π/4-DQPSK |
| QAM | 16QAM, 32QAM (cruz), 64QAM, 256QAM |
| APSK | 16APSK (4+12), 32APSK (4+12+16) |
| Frecuencia | 2FSK, 4FSK, MSK, GMSK |

**Señales idénticas: una sola clase** (decisión del usuario). La interfaz avisa de las equivalentes:

| Clase | También puede ser |
|---|---|
| BPSK | DBPSK |
| QPSK | DQPSK |
| OOK | 2ASK unipolar |
| MSK | OQPSK con pulso de medio seno; 2FSK de fase continua con h = 0,5 |

Para que las clases no se pisen:
- OQPSK nunca usa el pulso de medio seno (sería MSK).
- 2FSK usa h ∈ [0,7, 1,5]; 4FSK h ∈ [0,25, 1,0]; MSK h = 0,5 con pulso rectangular.
- GMSK usa h = 0,5 con BT ∈ {0,25, 0,3, 0,5}.

**Desconocidas nuevas.** 32QAM, 16APSK y 32APSK pasan a ser conocidas, así que las desconocidas cambian:

| Grupo | Modulaciones |
|---|---|
| Validación | 6PSK, 24QAM, HEX19, ANILLOS12, 3FSK |
| Prueba | 128QAM, 12PSK, 8QAM_ESTRELLA, ARB_A, ARB_B, 8FSK |

Las cifras de desconocidas de este catálogo no son comparables con las del catálogo original.

## Cómo se transmite

`src/banco/modulaciones.py` define `Modulacion` (nombre, familia, tipo, parámetros) y el catálogo digital.
`src/banco/senal.sintetizar` acepta una `Modulacion` o, como hasta ahora, un arreglo de puntos. El camino viejo no
cambia, así que el banco `base` sigue siendo reproducible.

| Tipo | Modulaciones | Transmisión |
|---|---|---|
| `lineal` | OOK, 4ASK, PAM, PSK, QAM, APSK | Puntos sorteados y los 6 generadores actuales. OOK y 4ASK no restan la media de sus puntos |
| `pi4dqpsk` | π/4-DQPSK | Saltos de fase de ±π/4 o ±3π/4 por símbolo, con los 6 generadores |
| `oqpsk` | OQPSK | El generador forma I y Q por separado; Q se retrasa medio símbolo (retardo fraccionario por FFT) |
| `cpm` | 2FSK, 4FSK, MSK, GMSK, 3FSK, 8FSK | Fase continua φ = π h Σ aₖ, con transmisor propio por generador (`src/banco/cpm.py` y `gr_bloques.py`) |

Los transmisores de fase continua de cada generador:

| Generador | Transmisor |
|---|---|
| `repo` | Exacto a sps entero |
| `procedural` | A 16 muestras por símbolo, interpolado con spline |
| `frecuencia` | Pulso de frecuencia filtrado en el dominio de la frecuencia a 48 muestras por símbolo |
| `analogico` | NRZ con filtro Butterworth o Bessel y VCO |
| `gr_polifase` | `frequency_modulator_fc` a 32 muestras por símbolo y `pfb_arb_resampler_ccf` |
| `gr_fir_mmse` | `interp_fir_filter_fff`, `frequency_modulator_fc` y `mmse_resampler_cc` |

Los parámetros sorteados de la modulación (h, BT) quedan en el manifiesto (`parametros_modulacion`), junto con la
`familia`.

`armar_banco.py --catalogo digital` arma el banco nuevo en `data/banco/digitales`.

## Aviso para la etapa 2

Las modulaciones de envolvente constante (FSK, MSK, GMSK) casi no dan línea cíclica en |x|². El normalizador actual
estima la tasa con esa línea, así que fallará con ellas. El enrutador tendrá que estimar la tasa con la frecuencia
instantánea.

## Criterios de aceptación

- Las 29 modulaciones (18 conocidas y 11 desconocidas) producen señales del largo pedido, finitas y reproducibles con
  cada generador y canal.
- OOK y 4ASK muestran una línea de portadora en la PSD; 4PAM no.
- 2FSK con h = 1 muestra dos tonos en ±h/(2·sps) ciclos por muestra.
- MSK tiene envolvente constante (canal ideal, sin ruido) y frecuencia instantánea en ±1/(4·sps).
- OQPSK: |x| nunca pasa por cero en canal ideal; QPSK sí se acerca a cero.
- π/4-DQPSK: los símbolos alternan entre dos conjuntos de 4 fases.
- El banco `base` se regenera bit a bit igual (el camino viejo no cambió).

---

# Etapa 2: enrutador de familias con especialistas

## Normalización sin tasa de símbolo

El normalizador de la cascada estima la tasa con la línea de |x|², que no existe en FSK/MSK/GMSK. Los especialistas
no la necesitan: `normalizar_banda` estima el centro y el ancho ocupado del espectro (cuantiles 2,5 % y 97,5 % de la
potencia por encima del piso de ruido), lleva el centro a cero y remuestrea para que ese ancho ocupe la mitad de la
tasa de muestreo. Así todas las señales llegan con la misma escala, sin saber su tasa ni su modulación.

## Vistas y especialistas (`src/familias/`)

| Especialista | Vista (tamaño fijo) | Qué debería aprender |
|---|---|---|
| Espectro | PSD de Welch en dB, 256 bins | Tonos de FSK, portadora de OOK/4ASK, forma del lóbulo |
| Potencias | Espectro de x², x⁴ y x⁸, 3 × 256 bins | Línea en x² (BPSK, PAM, ASK), x⁴ (QPSK, QAM), x⁸ (8PSK); ninguna en π/4-DQPSK ni FSK |
| Envolvente | Histograma de \|x\|/rms y autocorrelación de \|x\|, 2 × 64 | Envolvente constante frente a anillos; ritmo de símbolo |
| Frecuencia | Histograma de la frecuencia instantánea (cruda y suavizada), 2 × 128 | Niveles discretos de FSK; rampas de MSK/GMSK |

Cada especialista es una red convolucional 1-D pequeña, con la misma arquitectura para todos, que predice las 18
clases. La **fusión** es una regresión logística multinomial sobre las log-probabilidades de los cuatro especialistas
y entrega P(clase). P(familia) es la suma de las clases de cada familia.

## Datos y protocolo

| Uso | Banco | Contenido |
|---|---|---|
| Entrenamiento | `data/banco/digitales_entreno` | Solo las 18 clases, 4 por celda (13 824 archivos), semilla 2026112502 |
| Evaluación | `data/banco/digitales` | Banco de la etapa 1, con desconocidas |

Cada archivo aporta varias ventanas aleatorias como aumento de datos.

**Generalización a generadores no vistos:** tres pliegues, cada uno deja fuera un par de generadores. `repo` y
`gr_fir_mmse` van juntos porque con sps entero producen la misma señal; los otros pares son `procedural` con
`gr_polifase` y `frecuencia` con `analogico`.

En cada pliegue:
- Los especialistas se entrenan con el 80 % de los archivos de entrenamiento de los otros cuatro generadores.
- La fusión se ajusta con el 20 % restante.
- Se evalúa con los archivos del banco de evaluación que vienen de los dos generadores apartados.

**Métricas:**
- Exactitud de familia y de clase, global y por familia, canal y SNR.
- Cada especialista solo frente a la fusión.
- La familia asignada a las desconocidas: una 3FSK u 8FSK debería caer en «frecuencia», una 6PSK en «fase».

El modelo final se entrena con los seis generadores.

**Fuera de esta etapa:**
- La puerta de desconocidas y la lista conforme (etapa 3).
- La integración con la CNN de constelación y la interfaz.
