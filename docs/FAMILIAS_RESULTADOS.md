# Etapa 2: enrutador de familias con especialistas

Diseño en `docs/superpowers/specs/2026-11-25-modulaciones-digitales-design.md`. Código en `src/familias/` y
`experiments/exp_f1_especialistas.py`; resultados en `results/familias/resumen.json` y `filas_pliegues.json`; el modelo
final, entrenado con los seis generadores, en `results/familias/modelo/`.

## Resultado con generadores no vistos

Tres pliegues; cada uno deja fuera un par de generadores (`repo` + `gr_fir_mmse`, `procedural` + `gr_polifase`,
`frecuencia` + `analogico`). Son 6912 archivos conocidos del banco `data/banco/digitales`, con 6000 pasos de
entrenamiento por especialista.

| Exactitud | Fusión | Espectro | Potencias | Envolvente | Frecuencia |
|---|---|---|---|---|---|
| Familia, SNR ≥ 10 dB | **81,9 %** | 41,5 % | 69,2 % | 72,6 % | 65,5 % |
| Clase (18), SNR ≥ 10 dB | **65,5 %** | 25,0 % | 47,8 % | 45,0 % | 47,2 % |
| Familia, todas las SNR | 78,0 % | 40,4 % | 65,1 % | 67,5 % | 59,6 % |
| Clase, todas las SNR | 58,6 % | 23,0 % | 40,9 % | 37,4 % | 38,6 % |

La fusión supera al mejor especialista solo por 9 puntos en familia y 18 en clase: los especialistas se complementan.

**Familias a SNR ≥ 10 dB:**

| Familia | Exactitud |
|---|---|
| Amplitud | 95,5 % |
| Frecuencia | 93,7 % |
| Fase | 79,4 % |
| QAM | 78,7 % |
| APSK | 49,9 % |

**Clases a SNR ≥ 10 dB:**
- Las mejores son 2FSK (94,6 %), 4FSK (89,2 %), 4PAM, 4ASK, OOK y BPSK (81 % a 88 %).
- Las peores son las QAM densas: 64QAM (26,9 %) y 256QAM (21,5 %).

**Confusiones principales:**
- El orden de la QAM: 256 → 64, 256 → 16, 64 → 16.
- OOK con 4ASK.
- 16APSK con 32QAM.
- GMSK con MSK.
- QPSK con π/4-DQPSK.

Estas vistas resumen la señal: no ven la constelación, así que distinguir el orden de una QAM o separar APSK de QAM le
toca a la CNN de constelación de la cascada. El enrutador debe usarse para la familia y como prior, no como clasificador
final.

**Por canal:**
- Los canales con multitrayecto son el punto débil: `gr_multitrayecto` 28,7 %, `ecos_largos` 36,2 %, `mp_largo` 38,4 %,
  TDL 42–43 %.
- Ideal, IQ/DC, imperfecciones de GNU Radio y cuantización quedan entre 86 % y 89 %.

**Desconocidas (familia asignada, SNR ≥ 10 dB):**

| Desconocida | Familia asignada |
|---|---|
| 3FSK | frecuencia, 93,5 % |
| 8FSK | frecuencia, 97,2 % |
| 6PSK | fase, 60,1 % |
| 12PSK | fase, 57,7 % |
| 128QAM | QAM, 72,0 % |
| 24QAM | QAM, 79,2 % |
| HEX19 | se reparte entre QAM y APSK |
| ANILLOS12 | se reparte entre QAM y APSK |
| 8QAM_ESTRELLA | se reparte entre fase y QAM |
| ARB_B | se reparte entre fase y QAM |

Las desconocidas caen en la familia que les corresponde, lo que sirve para decir «parece de la familia X, pero no es
ninguna de las conocidas».

## Unidades nuevas (superteam, 2026-11-26)

| Unidad | Archivo | Qué hace | Estado |
|---|---|---|---|
| U7 | `src/familias/tasa.py` | Tasa de símbolo ciega también para envolvente constante (FSK, MSK, GMSK), con tres detectores cíclicos (envolvente, \|Δf\| y f²) | Contrato revisado; segundo intento implementado por Claude |
| U8 | `src/familias/tonos.py` | Cuenta los tonos de una FSK y estima el índice de modulación h con el histograma de la frecuencia instantánea en los tramos estables | Contrato revisado; aceptada al segundo intento de agy |
| U9 | `src/familias/conforme.py` | Lista conforme «también puede ser» (LAC) y salida final con equivalencias «misma señal que» | Aceptada al primer intento; criterio revisado |

**U7, tasa ciega.** La regla es:
1. Filtrar a la banda ocupada.
2. Aplicar los tres detectores.
3. Medir la prominencia de cada detector contra la mediana de su banda.
4. Entre los detectores fuertes, tomar la frecuencia más baja, para no caer en armónicos.

Resultados a SNR 20:
- MSK, 2FSK, 4FSK y QPSK a 4, 8 y 16 muestras por símbolo dan error < 2 % en los 36 casos.
- GMSK acierta con BT 0,5 y falla en los 3 casos con BT 0,3 (la semilla 1 sortea ese BT): estima de 26 a 37 muestras
  por símbolo. Con BT bajo, el pulso gaussiano borra casi todo el rasgo cíclico.

El puntaje no sirve para detectar si hay señal: el ruido puro da una mediana de 4,1 y el GMSK más débil 5,6.

**U8, tonos.**
- El primer intento contaba 5 tonos en 4FSK: las transiciones entre tonos llenaban el histograma y creaban un pico
  falso cerca de 0.
- El contrato se corrigió para usar solo los tramos con varianza local por debajo de la mediana.
- Con 16 muestras por símbolo, SNR 30 y semillas 0 a 2:

| Señal | Tonos | h estimado |
|---|---|---|
| 2FSK h = 1 | 2 | 0,975 |
| 4FSK h = 0,5 | 4 | 0,500 |
| 3FSK h = 0,8 | 3 | 0,788 |
| MSK | 2 | 0,475 |

**U9, conforme.**
- La cobertura queda a ±0,02 de max(1 − α, exactitud del argmax).
- El conjunto siempre incluye el argmax, así que la cobertura no puede bajar de esa exactitud.

Pruebas en `tests/test_familias_unidades.py`.

## Integración con la constelación (2026-11-27)

`experiments/exp_f2_integracion.py`, resultados en `results/familias/integracion/resumen.json` y el paquete en
`results/familias/integracion/paquete_integrado.joblib` (motor «Familias + constelación» de la interfaz, el de defecto).

**Cómo decide.** P(modulación) = P(familia) × P(modulación | familia).
- P(familia) es la del enrutador.
- Dentro de las familias de constelación, P(modulación | familia) es el producto enrutador × decisor de la cascada,
  renormalizado en la familia.
- La probabilidad de desconocida viene de la puerta del decisor.
- En la familia de frecuencia decide el enrutador. El contador de tonos marca desconocida si ve 3 tonos o más,
  distintos de 4, con puntaje ≥ 0,8.

**Datos.**
- El decisor aprendió las 14 modulaciones de constelación (OOK … 32APSK) con las huellas de `digitales_entreno`, sin
  reentrenar la CNN.
- El banco `digitales` se parte en dos:
  - Calibración: los archivos con índice 0 del catálogo y las desconocidas de validación.
  - Evaluación: los archivos con índice 1 y las desconocidas de prueba (128QAM, 12PSK, 8FSK, 8QAM estrella, ARB_A,
    ARB_B).
- Los generadores son los mismos del entrenamiento: estos números no son comparables con los de la tabla de arriba, que
  deja fuera pares de generadores.

**Decisiones tomadas con la partición de calibración.**

| Opción dentro de la familia | Exactitud dentro de la familia | Exactitud en las 14 clases |
|---|---|---|
| Enrutador solo | 74,9 % | 66,1 % |
| Decisor solo | 65,1 % | 57,6 % |
| Producto | **78,3 %** | **73,1 %** |

- El producto gana: la constelación sube APSK de 81 % a 88,5 % y QAM de 47 % a 56 %.
- Tonos: sin el corte de puntaje, la regla marcaba como desconocida el 40 % de las FSK conocidas. Con puntaje ≥ 0,8
  marca el 1,7 % y atrapa el 42,7 % de las 3FSK.
- Puerta: el intercepto se desplazó −1,7. Es el mayor desplazamiento que deja como mucho un 10 % de conocidas marcadas
  como desconocidas a SNR ≥ 10 dB.
- Umbral: 0,31, el mayor que deja como mucho un 10 % de conocidas sin decidir.
  - El criterio de la cascada (85 % de exactitud entre las aceptadas) daba un umbral de 0,89 y dejaba sin decidir el
    65 % de las conocidas.
  - Con 18 clases, la duda debería expresarse con la lista «también puede ser» de la etapa 3, no con un umbral alto.

**Resultado en la partición de evaluación, SNR ≥ 10 dB:**

| | Fusión | Fusión sin umbral | Enrutador solo | Cascada original (sus 6 clases) |
|---|---|---|---|---|
| Conocidas correctas | 69,7 % | 70,7 % | 69,6 % | 41,9 % |
| Conocidas sin decidir | 8,1 % | 0 % | 0 % | 47,5 % |
| Conocidas marcadas desconocidas | 3,9 % | 6,8 % | 0 % | 10,1 % |
| Desconocidas detectadas | 30,2 % | 36,9 % | 0 % | 43,6 % |
| Desconocidas no aceptadas como conocida | 48,1 % | 36,9 % | 0 % | 95,1 % |

- En las 6 clases que la cascada ya conocía, la fusión acierta 66,2 % frente al 41,9 % de la cascada original.
- En las 18 clases empata con el enrutador, pero agrega lo que el enrutador no tiene: puede decir «desconocida».
- Por familia, fusión frente a enrutador:

| Familia | Fusión | Enrutador |
|---|---|---|
| Amplitud | 88,9 % | 89,4 % |
| Frecuencia | 85,1 % | 86,1 % |
| Fase | 74,0 % | 73,1 % |
| APSK | 58,4 % | 63,0 % |
| QAM | 37,7 % | 34,7 % |

  Con umbral, las «sin decidir» cuentan como error.
- La 16QAM mejora de 38,9 % a 55,6 %. La 256QAM empeora de 36,7 % a 28,3 % por el umbral y la puerta.
- **Desconocidas:**

| Desconocida | Detectada | No aceptada como conocida |
|---|---|---|
| ARB_A | 63 % | 80 % |
| ARB_B | 57 % | 74 % |
| 8QAM estrella | 24 % | 42 % |
| 8FSK | 19 % | 19 % |
| 128QAM | 14 % | 56 % |
| 12PSK | 4 % | 17 % |

  Con 14 clases inscritas, una 12PSK o una 128QAM se parecen demasiado a 8PSK o 256QAM: la puerta casi no las separa.
- **Por canal:** de 91 % en ideal, LTE-EPA y deriva de portadora a 28 % en `mp_largo` y 36 % en `gr_multitrayecto`.

**Interfaz.**
- El motor nuevo devuelve `enrutador` con:
  - las familias;
  - la clase que vota cada especialista;
  - la rama (constelación o frecuencia);
  - en la rama de frecuencia, los tonos, h, muestras por símbolo y si el conteo es fiable.
- La tarjeta «Enrutador de familias» lo muestra entre la constelación y la salida de la CNN.
- El banco de ejemplos pasó a `data/banco/digitales`.
- Código:
  - Fusión: `src/familias/integracion.py` (U10, agy; contrato revisado al producto de expertos).
  - Tarjeta: `pintarEnrutador` en `src/interfaz/web/app.js` (U11, agy).
  - Pegamento: `src/interfaz/integrado.py`.

## Etapa 3: «también puede ser» y «misma señal que» (2026-11-27)

El motor integrado ahora devuelve, junto a la decisión:
- `tambien_puede_ser`: las otras modulaciones de la lista conforme, con su probabilidad.
- `misma_senal_que`: las equivalencias del catálogo (`EQUIVALENTES` en `src/banco/modulaciones.py`):
  - BPSK = DBPSK;
  - QPSK = DQPSK;
  - OOK = 2ASK unipolar;
  - MSK = OQPSK con pulso de medio seno = 2FSK de fase continua con h = 0,5.
- `mas_probable`, que se muestra cuando la decisión queda «sin decidir».

La interfaz lo escribe bajo la modulación detectada (`pintarAlternativas` en `app.js`, unidad U12 de agy).

**Cómo se calcula la lista.**
- Es una lista conforme LAC (`src/familias/conforme.py`, U9) sobre las 19 salidas de la fusión. Incluye «Desconocida»:
  para una señal no vista, la lista puede decir «también puede ser: desconocida».
- Se calibra con la partición de calibración en todas las SNR, con α = 0,1.
- El corte queda en 0,967: entra toda opción con probabilidad ≥ 3,3 %.
- Calibrar solo con SNR ≥ 10 dB daba listas algo más cortas (3,2 opciones), pero con 84 % de cobertura a SNR baja.

**Resultado en la partición de evaluación:**

| | La respuesta correcta está en la lista | Opciones de media | Una sola opción |
|---|---|---|---|
| SNR < 10 dB | 88,1 % | 4,9 | 10 % |
| SNR ≥ 10 dB | 90,3 % | 3,6 | 20 % |
| Conocidas, SNR ≥ 10 dB | 93,3 % | 3,2 | 23 % |
| Desconocidas, SNR ≥ 10 dB («Desconocida» en la lista) | 85,9 % | 4,2 | 16 % |

La cobertura cumple el 90 % pedido. Las listas son largas porque la fusión acierta cerca del 70 %: cuando duda, lo dice,
en vez de esconderlo detrás de una sola respuesta. Con canales difíciles (TDL Rician, multitrayecto) la lista llega a 8 o
9 opciones.

## Etapa 4: especialistas dentro de cada familia (2026-11-27)

`experiments/exp_f3_especialistas_familia.py`. Los modelos, `info.json` y `medicion.json` están en
`results/familias/especialistas_familia/`.

**Qué mira el especialista.**
- Toma los símbolos que recupera la cascada robusta, hasta 2048 por archivo.
- Con ellos arma cuatro histogramas finos de 256 casillas (`src/familias/detalle.py`, U13, agy): I, Q, |s| y fase.
- Antes de los histogramas normaliza la potencia y gira la constelación con la cuarta potencia. Así la rejilla de una
  QAM cae sobre los ejes y cada nivel es un pico.
- La red es la misma CNN 1-D de los especialistas de familia (`Especialista`).
- Se entrena con `digitales_entreno`, sin frecuencia. Cada archivo aporta 4 ventanas de largo aleatorio (≥ 256
  símbolos), para que aprenda también con capturas cortas.

**Medición dentro de la familia** (partición de calibración, SNR ≥ 10 dB):

| Grupo | n | Enrutador | Decisor | Especialista | Enr. × dec. | Enr. × dec. × esp. |
|---|---|---|---|---|---|---|
| Amplitud | 387 | **92,0 %** | 74,4 % | 76,7 % | **92,0 %** | 90,2 % |
| Fase | 637 | **84,3 %** | 56,7 % | 54,3 % | 83,7 % | 79,3 % |
| QAM | 509 | 47,0 % | 56,6 % | 50,1 % | 55,8 % | **58,9 %** |
| APSK | 269 | 81,0 % | 88,1 % | 85,5 % | 88,5 % | **88,9 %** |
| QAM + APSK juntas | 778 | 47,6 % | 60,2 % | 56,0 % | 59,1 % | **64,1 %** |

- El especialista solo vale la pena en QAM y APSK. En amplitud y fase el enrutador ya separa las clases, y el producto
  con el especialista empeora. En frecuencia no hay constelación que mirar.
- Lo mejor es tratar QAM y APSK como un solo grupo. El enrutador las confunde entre sí: un 16APSK cae en «QAM» con
  frecuencia. El especialista ve los anillos de la APSK y lo corrige.
- **Límite de la física.**
  - En señales limpias el especialista acierta el 74 % de las QAM, y entre 83 % y 88 % con 1024 símbolos o más.
  - 64QAM y 256QAM necesitan más SNR y más símbolos de los que tiene buena parte del banco: entre niveles vecinos de
    una 256QAM hay 1/15 de la amplitud.

**Cómo entra en la fusión.**
- `fusionar` recibe `grupos=[("qam", "apsk")]` y `p_especialista`. Dentro del grupo, P(m | grupo) es enrutador ×
  decisor × especialista, renormalizado.
- P(grupo) es la suma de P(qam) y P(apsk) del enrutador.
- Las demás familias siguen igual.
- Con el especialista se recalibró todo con la partición de calibración:
  - Puerta: el desplazamiento pasó a −1,5.
  - Umbral: pasó a 0,36.
  - Corte conforme: pasó a 0,975.

**Resultado en la partición de evaluación, SNR ≥ 10 dB:**

| | Fusión | Fusión sin umbral | Sin especialista, sin umbral | Enrutador solo |
|---|---|---|---|---|
| Conocidas correctas | **71,2 %** | **73,3 %** | 69,9 % | 69,6 % |
| Conocidas sin decidir | 8,9 % | 0 % | 0 % | 0 % |
| Conocidas marcadas desconocidas | 4,1 % | 6,9 % | 9,1 % | 0 % |
| Desconocidas detectadas | 30,2 % | 37,5 % | 42,6 % | 0 % |
| Desconocidas no aceptadas como conocida | 47,2 % | 37,5 % | 42,6 % | 0 % |

La columna «sin especialista» usa la misma puerta recalibrada; por eso detecta más desconocidas que en la sección de
integración.

| Familia | Antes | Con especialista | Enrutador |
|---|---|---|---|
| APSK | 58,4 % | **66,4 %** | 63,0 % |
| QAM | 37,7 % | **44,6 %** | 34,7 % |
| Amplitud | 88,9 % | 88,4 % | 89,4 % |
| Fase | 74,0 % | 71,7 % | 73,1 % |
| Frecuencia | 85,1 % | 84,9 % | 86,1 % |

- Por clase: 16QAM 58,7 %, 32QAM 47,5 %, 64QAM 27,1 %, 256QAM 42,5 % (antes 28,3 %), 16APSK 63,6 %, 32APSK 69,2 %.
- En las 6 clases de la cascada original, la fusión sube a 65,9 % (la cascada original daba 41,9 %).
- La lista «también puede ser» cubre el 90,5 % a SNR ≥ 10 dB con 3,5 opciones de media (antes 3,6).
- **Lo que cuesta.**
  - El especialista solo conoce 6 clases: ante una 128QAM o una 8QAM estrella elige la más parecida con mucha seguridad.
  - Eso le quita masa a la puerta: sin umbral, las desconocidas detectadas bajan de 42,6 % a 37,5 %.
  - La 128QAM detectada bajó a 11,7 %; la 8QAM estrella está en 24,1 %.

**Interfaz.**
- La tarjeta del enrutador suma la fila «Constelación (QAM/APSK)» cuando la familia más probable es QAM o APSK.
- Ejemplo: un 16APSK del banco en canal ideal.
  - El enrutador duda: QAM 51,9 %, APSK 46,3 %.
  - El especialista vota 16APSK con 89,5 %.
  - La fusión responde 16APSK con 95,2 %.

## Etapa 5: desconocidas parecidas a una conocida (2026-11-27)

`experiments/exp_f4_desconocidas.py` y `src/familias/ajuste.py` (U14, agy). Mediciones y selección en
`results/familias/desconocidas/`.

**La idea.** Antes de responder una clase, se comprueba que los símbolos recuperados caben en su constelación ideal.
Es física pura y no se entrena con desconocidas.

**Cómo se mide**, para cada clase de constelación (14 plantillas; PI4DQPSK se ve como 8PSK):
- **Error contra la plantilla.**
  - Se buscan el giro y la ganancia que mejor la ajustan: 48 giros más 3 pasos de mínimos cuadrados.
  - El error se divide por el de un ajuste libre, k-means con 64 puntos. Así no hace falta conocer la SNR.
  - Si los símbolos forman nubes apretadas que no caen sobre la plantilla, la razón se dispara. Es el caso de una
    12PSK contra 8PSK.
- **Ocupación.** Es el número efectivo de puntos de la plantilla que se usan (entropía de las cuentas). Una 128QAM
  cae sobre parte de la rejilla de 256QAM y deja vacías las esquinas.

**Dos intentos fallidos con el prototipo:**
- Tomar «la plantilla que mejor ajusta» no sirve: la de 256QAM ajusta cualquier señal, y la plantilla correcta ganaba
  solo el 8 % de las veces.
- Comparar con un k-means del doble de puntos que la plantilla tampoco: BPSK y OOK «pasaban» con cualquier señal.

**Calibración.**
- Umbrales de cada clase:
  - Salen de `digitales_entreno`: unos 400 archivos por clase a SNR ≥ 10 dB.
  - Con los ~100 archivos de la partición de calibración, el cuantil del 2 % era ruidoso.
  - Una clase no cabe si su razón supera el cuantil 1 − β de su propia clase, o su ocupación queda bajo el cuantil β.
- Fusión:
  - Una clase que no cabe conserva un peso w de su masa.
  - El resto pasa a «Desconocida» (`fusionar(..., consistentes=...)`, revisión U10d, agy).
- β y w:
  - Se eligieron con la partición de calibración y la misma regla de la puerta: como mucho un 10 % de conocidas
    marcadas desconocidas a SNR ≥ 10 dB.
  - Quedaron β = 0,02 y w = 0,5. Con β = 0,1 se detectaba más, pero marcaba el 14 % de las conocidas.
  - La puerta pasó a −1,6 y el umbral a 0,35.

**Selección en la partición de calibración** (SNR ≥ 10 dB; desconocidas de validación):

| | Sin ajuste | β 0,02, w 0,5 |
|---|---|---|
| Conocidas correctas | 70,8 % | 70,3 % |
| Conocidas marcadas desconocidas | 4,8 % | 5,8 % |
| Desconocidas detectadas | 30,6 % | 44,3 % |
| 6PSK | 3,5 % | 41,9 % |
| 24QAM | 23,8 % | 37,7 % |

**Resultado en la partición de evaluación**, SNR ≥ 10 dB. Es la primera vez que se miran 12PSK y 128QAM:

| | Etapa 4 | Con ajuste | Con ajuste, sin umbral |
|---|---|---|---|
| Conocidas correctas | 71,2 % | 70,4 % | 72,5 % |
| Conocidas sin decidir | 8,9 % | 8,7 % | 0 % |
| Conocidas marcadas desconocidas | 4,1 % | 5,1 % | 8,0 % |
| Desconocidas detectadas | 30,2 % | **38,4 %** | 44,4 % |
| Desconocidas no aceptadas como conocida | 47,2 % | **52,1 %** | 44,4 % |

| Desconocida | Etapa 4 | Con ajuste |
|---|---|---|
| 12PSK | 5,0 % | **25,4 %** |
| 8QAM estrella | 24,1 % | **48,2 %** |
| 128QAM | 11,7 % | 10,5 % |
| ARB_A | 64,2 % | 66,4 % |
| ARB_B | 56,5 % | 58,9 % |
| 8FSK | 19,0 % | 19,4 % |

- **128QAM** no mejora. En las sintéticas a 30 dB la ocupación la separa (0,68 frente a 0,94 de una 256QAM), pero a
  15–20 dB queda en 0,85–0,88 frente a 0,93. Las nubes de una 256QAM ya se tocan y la rejilla ocupada no se distingue.
  Es el mismo límite físico que tiene 256QAM frente a 64QAM.
- **Lo que cuesta.**
  - En conocidas se pierde 0,8 puntos. Casi todo es OOK, de 85,8 % a 78,7 %: 10 de sus 92 archivos de evaluación
    no caben, frente al ~4 % esperado (4,3 % en entreno y 3,9 % en calibración).
  - Los que fallan están repartidos entre generadores y canales; con 92 archivos parece azar. No se ajustó nada mirando
    la evaluación.
  - Las demás clases pierden de 0 a 2 puntos.
- La lista «también puede ser» cubre el 90,7 % con 3,4 opciones de media.
- **Tiempo:** la prueba agrega unos 0,8 s por señal en la interfaz (15 plantillas y un k-means).

**Interfaz.** Debajo de la decisión aparece una nota cuando una de las tres clases más probables no cabe. Por ejemplo,
con una 12PSK de `gr_polifase` con deriva de reloj, la respuesta es «Desconocida» y la nota dice: «Los símbolos no caben
en la constelación ideal de 8PSK, 32QAM, 64QAM».

## Falta
- 128QAM frente a 256QAM: haría falta más SNR o más símbolos; la prueba no puede inventar la separación.
- GMSK frente a MSK: U8 no separa BT 0,3.
- El multitrayecto sigue débil, como en la cascada.
