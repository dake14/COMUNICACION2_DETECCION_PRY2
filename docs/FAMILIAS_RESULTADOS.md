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

## Falta

- Integrar el enrutador con la CNN de constelación y con la interfaz.
- Etapa 3: calibrar la lista conforme con las probabilidades reales y mostrar «también puede ser» y «misma señal que» en
  la salida.
- Especialistas dentro de cada familia: el orden de QAM, APSK frente a QAM, y GMSK frente a MSK con U8.
- El multitrayecto sigue débil, como en la cascada.
