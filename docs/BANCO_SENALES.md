# Banco de señales

Genera señales I/Q cruzando transmisores, canales, modulaciones y configuraciones. Cada archivo queda registrado en
`manifiesto.jsonl` con su generador, canal, modulación, configuración y parámetros sorteados. Eso permite apartar
después un generador o un canal completo.

## Generadores

| Nombre | Implementación | Pulsos | Tasa fraccionaria |
|---|---|---|---|
| `repo` | Conformador original del proyecto | RRC | No |
| `procedural` | Convolución a 8 sps e interpolación cúbica | RRC, RC, gaussiano, medio seno, Kaiser | Sí |
| `frecuencia` | Pulso aplicado en frecuencia y remuestreo polifásico | RRC, gaussiano, coseno truncado | Sí |
| `analogico` | Retención de orden cero y filtro Butterworth o Bessel | NRZ filtrado | Sí |
| `gr_polifase` | GNU Radio, `pfb_arb_resampler_ccf` | RRC, gaussiano | Sí |
| `gr_fir_mmse` | GNU Radio, FIR interpolador y remuestreador MMSE | RRC, gaussiano | Sí |

## Canales

| Grupo | Canales |
|---|---|
| Referencia | `ideal`, `proyecto` |
| Multitrayecto fijo | `mp_corto`, `mp_largo`, `gr_multitrayecto` |
| Desvanecimiento | `plano_lento`, `tdl_rayleigh`, `tdl_rician`, `gr_selectivo`, `gr_dinamico` |
| Sincronía y fase | `ruido_fase`, `deriva_portadora`, `deriva_reloj` |
| Hardware | `iq_dc`, `amplificador`, `cuantizacion`, `receptor_barato`, `gr_imperfecciones` |
| Otras señales o ruido | `interferencia`, `impulsivo` |

## Configuraciones

Cada archivo sortea:

- Muestras por símbolo: 3, 4, 5, 6, 8, 10 y 12, más 3,5, 5,5, 6,4, 7,3 y 9,6 si el generador admite tasa fraccionaria.
- Factor β: 0,15, 0,2, 0,25, 0,35, 0,5 y 0,7.
- Longitud del pulso: 6, 8, 10, 12 y 16 símbolos.
- Símbolos por archivo: 256, 512, 1024 y 2048.
- SNR: 0, 5, 10, 15, 20 y 30 dB.
- Desvío de portadora, fase y ganancia.

## Uso

Con todos los generadores y canales, incluidos los de GNU Radio:

```bash
C:/Users/David/radioconda/python.exe -m experiments.armar_banco --salida data/banco/base --por-celda 4 --trabajos 8
```

Se pueden limitar con `--generadores`, `--canales` y `--modulaciones`, separados por comas. Con el `.venv` del proyecto
solo aparecen los componentes que no necesitan GNU Radio.

Cada valor de `--por-celda` son 2160 archivos: 6 generadores, 20 canales y 18 modulaciones.

## Nota sobre el pulso de medio seno

Con este pulso, |x|² no tiene línea a la tasa de símbolo, aunque la señal es correcta. La tasa sí aparece en la derivada.
Es un caso difícil a propósito para cualquier estimador basado solo en |x|².
