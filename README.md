# LABORATORIO 5
### OBJETIVO
Identificar cambios en el balance autonómico mediante análisis temporal de la
variabilidad de la frecuencia cardíaca (HRV).

# PARTE A

<img width="385" height="571" alt="image" src="https://github.com/user-attachments/assets/e1c437f6-2b18-4f31-b480-a99615fa7f83" />

Se realizó una investigación teórica, sobre los siguientes temas:

**1. Actividad simpática y parasimpática del sistema nervioso autónomo**


El sistema nervioso autónomo (SNA) es la parte del sistema nervioso encargada de regular de manera involuntaria funciones vitales como la frecuencia cardíaca, la respiración, la digestión, la actividad glandular y el diámetro pupilar. Su función principal es mantener la homeostasis del organismo y permitir que el cuerpo responda adecuadamente a las demandas internas y externas. El SNA se divide en dos ramas que actúan de forma complementaria y muchas veces opuesta: el sistema nervioso simpático y el sistema nervioso parasimpático.


El sistema nervioso simpático está asociado con la respuesta de “lucha o huida”, un conjunto de reacciones fisiológicas que se activan ante situaciones de estrés, peligro o aumento de la exigencia física. Cuando se activa esta rama, el organismo se prepara para la acción: aumenta la frecuencia y la fuerza de contracción del corazón, se dilatan las pupilas y los bronquios para mejorar la entrada de luz y aire, se movilizan las reservas de glucosa para obtener energía rápida y se redirige el flujo sanguíneo hacia los músculos esqueléticos. Al mismo tiempo, se inhiben funciones no prioritarias como la digestión, disminuyendo la motilidad gastrointestinal y las secreciones digestivas. También se produce vasoconstricción en piel y vísceras, sudoración y la liberación de adrenalina y noradrenalina desde la médula suprarrenal. En este sistema, la noradrenalina es el neurotransmisor predominante en las sinapsis posganglionares, mientras que la acetilcolina participa principalmente en las sinapsis preganglionares.


Por su parte, el sistema nervioso parasimpático promueve la respuesta de “reposo y digestión”, predominante en condiciones de calma y recuperación. Su activación favorece funciones destinadas a conservar y restaurar la energía del organismo. Entre sus efectos se encuentran la disminución de la frecuencia cardíaca, la constricción pupilar, la broncoconstricción y la estimulación de la actividad del tracto gastrointestinal, lo que incluye aumento de la motilidad, secreciones digestivas y peristaltismo. Este sistema también estimula funciones como la salivación, la micción, la defecación y la actividad de glándulas lagrimales. A diferencia del simpático, el neurotransmisor predominante en las sinapsis tanto preganglionares como posganglionares es la acetilcolina, que actúa sobre receptores nicotínicos y muscarínicos.


Ambos sistemas, simpático y parasimpático, trabajan de manera coordinada para mantener el equilibrio del organismo. Aunque sus acciones suelen ser opuestas, no actúan de forma aislada; más bien, el estado fisiológico depende del predominio relativo de uno u otro según las necesidades del momento. En situación de reposo suele dominar el parasimpático, mientras que en momentos de estrés o actividad física predomina el simpático. Esta interacción permite respuestas rápidas, adaptativas y eficientes, garantizando el adecuado funcionamiento de los órganos y sistemas del cuerpo.

**2. Efecto de la actividad simpática y parasimpática en la frecuencia cardíaca**


La frecuencia cardíaca está regulada por el equilibrio entre la actividad simpática y parasimpática del sistema nervioso autónomo. La actividad simpática aumenta la frecuencia cardíaca mediante la liberación de noradrenalina y adrenalina, que actúan sobre receptores β1 del corazón, acelerando la despolarización del nodo sinoauricular y mejorando la conducción eléctrica, lo que prepara al organismo para situaciones de estrés, alarma o esfuerzo físico. En contraste, la actividad parasimpática, principalmente a través del nervio vago y la liberación de acetilcolina sobre receptores muscarínicos M2, disminuye la frecuencia cardíaca al reducir la velocidad de despolarización del nodo SA y enlentecer la conducción en el nodo AV, predominando en condiciones de reposo y favoreciendo la conservación de energía. Así, el ritmo cardíaco final es el resultado del balance dinámico entre ambas ramas, que ajustan el funcionamiento del corazón según las necesidades del organismo.

**3. Variabilidad de la frecuencia cardíaca (HRV) obtenida a partir de la señal electrocardiográfica (ECG)**


La variabilidad de la frecuencia cardíaca (HRV) es una medida que refleja las fluctuaciones naturales en el intervalo de tiempo entre un latido cardíaco y el siguiente, conocidas como intervalos R-R, las cuales se obtienen a partir del análisis de la señal electrocardiográfica (ECG). Estas variaciones no son aleatorias, sino que representan la interacción dinámica entre las ramas simpática y parasimpática del sistema nervioso autónomo. Para calcular la HRV, primero se identifican los picos R en el ECG —los puntos de mayor amplitud del complejo QRS— y se determina la duración entre cada par de picos consecutivos; la serie de estos intervalos permite aplicar análisis temporales (como SDNN o RMSSD), frecuenciales (como las bandas LF y HF) o no lineales. Una HRV elevada indica un sistema autónomo flexible y eficiente, con predominio parasimpático y buena capacidad de adaptación fisiológica, mientras que valores bajos suelen asociarse con estrés, fatiga, disfunción autonómica o estados patológicos. En conjunto, la HRV derivada del ECG es una herramienta no invasiva, sensible y ampliamente utilizada para evaluar la regulación autonómica del corazón y el estado general del organismo.

**4. Diagrama de Poincaré como herramienta de análisis de la serie R-R.**


El diagrama de Poincaré es una herramienta gráfica utilizada para analizar la variabilidad de la frecuencia cardíaca (HRV) a partir de la serie de intervalos R-R obtenidos del electrocardiograma. Consiste en representar cada intervalo R-R en función del siguiente, es decir, colocar cada valor  RRn+1  en el eje horizontal y el intervalo siguiente  RRn+1  en el eje vertical. Al hacerlo, se genera una nube de puntos cuya forma y dispersión permiten visualizar de manera sencilla la dinámica del sistema nervioso autónomo. En condiciones fisiológicas normales, el gráfico adopta una forma elipsoidal cuyo ancho transversal, conocido como SD1, está relacionado con la variabilidad instantánea de los latidos y refleja principalmente la actividad parasimpática. Por otro lado, la longitud del eje mayor, conocida como SD2, representa la variabilidad a más largo plazo, influida tanto por modulaciones simpáticas como parasimpáticas. Un diagrama amplio y disperso suele asociarse con una regulación autonómica saludable y flexible, mientras que un patrón estrecho, lineal o muy compacto indica una disminución de la variabilidad, lo cual puede relacionarse con estrés, fatiga o disfunción autonómica. Gracias a su capacidad para mostrar patrones no lineales y ofrecer una interpretación visual rápida, el diagrama de Poincaré se considera un complemento valioso para el análisis tradicional de la HRV.

+ **Posteriormente se adquirió la señal ECG**

Se adquirió la señal por medio del DAQ

```python
import pandas as pd
import io

df = pd.read_csv(io.BytesIO(uploaded['Señal_ECG (2).txt']),
                 sep=r'\s+',
                 header=0)

df.head()
```
| t [s]   | Voltaje [V] |
|---------|--------------|
| 0.0000  | 2.334708     |
| 0.0005  | 2.528737     |
| 0.0010  | 2.202068     |
| 0.0015  | 2.168427     |
| 0.0020  | 2.459448     |

+ **Gráfica señal adquirida**

```python
import matplotlib.pyplot as plt
  import numpy as np

  plt.figure(figsize=(8, 5))  

  plt.plot(df['t[s]'], df['voltaje[V]'], linewidth=0.5)

  plt.xlim(0, 240)            # tiempo [s]
  plt.ylim(-4, 4)             # voltaje [V]

  plt.xticks(np.arange(0, 241, 50))
  plt.yticks(np.arange(-4, 5, 1))


  plt.xlabel('Tiempo [s]')
  plt.ylabel('Voltaje [V]')
  plt.title('Señal ECG')

  # Cuadrícula
  plt.grid(True)

  ax = plt.gca()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.tight_layout()
  plt.show()
  
  ```
  <img width="630" height="392" alt="image" src="https://github.com/user-attachments/assets/80db840b-5d96-44af-a310-ec82b5c58c3c" />

  # PARTE B

  <img width="311" height="653" alt="image" src="https://github.com/user-attachments/assets/ffc47a48-2b6f-4ac0-addf-4711b1b2863f" />

Pre - procesamiento de la señal
Aplicar los filtros digitales necesarios para eliminar el ruido de la señal, demostrando su diseño.
- Diseñar un filtro IIR de acuerdo con los parámetros de la señal,
- Obtener la ecuación en diferencias del filtro,
- Implementar el filtro a la señal obtenida asumiendo parámetros iniciales en 0.

```python
# Leer el archivo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo 
df = pd.read_csv("Señal_ECG (2).txt", sep=r"\s+", header=0)

# Columnas: tiempo y voltaje
t = df["t[s]"].values.astype(float)
x = df["voltaje[V]"].values.astype(float)

print("Muestras:", len(x))
print("Tiempo total [s]:", t[-1] - t[0])
print("Voltaje min/max:", x.min(), x.max())

plt.figure(figsize=(12,4))
plt.plot(t, x, alpha=0.7)
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title("ECG inicial (señal cruda)")
plt.grid(True)
plt.tight_layout()
plt.show()
```
**Este código carga la señal ECG desde un archivo, extrae sus datos y grafica el voltaje en función del tiempo.**
+ *Muestras: 480000*
+ *Tiempo total [s]: 239.9995*
+ *Voltaje min/max: 1.245738154976885 3.5*

<img width="952" height="312" alt="image" src="https://github.com/user-attachments/assets/d7320795-d7d5-4042-a7ab-4a0e5621fdcf" />

+ **Diseño Filtro IIR**
```python
# Diseñar el filtro IIR, pasa–banda para ECG
# se filtra aprox. 0.5 – 40 Hz.

from scipy.signal import butter, sosfilt

# Diseño del mismo filtro pero en forma SOS
sos = butter(orden, [low, high], btype="bandpass", output="sos")
print("Matriz SOS:\n", sos)

# Filtrado con condiciones iniciales = 0
ecg_filtrado = sosfilt(sos, x)

plt.figure(figsize=(12,4))
plt.plot(t, x, label="ECG inicial", alpha=0.4)
plt.plot(t, ecg_filtrado, label="ECG filtrado", linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.title("ECG antes y después del filtro IIR Butterworth 0.5–40 Hz")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```
Matriz en formato SOS que contiene los coeficientes de cada sección biquad del filtro digital implementado.

| b0           | b1           | b2           | a0 | a1           | a2           |
|--------------|--------------|--------------|----|--------------|--------------|
| 1.26657730e-05 | 2.53315460e-05 | 1.26657730e-05 | 1  | -1.78354137e+00 | 7.97115047e-01 |
| 1.00000000e+00 | 2.00000000e+00 | 1.00000000e+00 | 1  | -1.89566508e+00 | 9.10625808e-01 |
| 1.00000000e+00 | -2.00000000e+00 | 1.00000000e+00 | 1  | -1.99704749e+00 | 9.97050061e-01 |
| 1.00000000e+00 | -2.00000000e+00 | 1.00000000e+00 | 1  | -1.99881745e+00 | 9.98819938e-01 |

<img width="948" height="310" alt="image" src="https://github.com/user-attachments/assets/fbf1883c-8dbd-4624-bf89-d38aaf6f8663" />

+ **Dividir la señal filtrada en dos segmentos de señal con duración de 2 minutoscada uno**
 ```python
# Duración de cada segmento (2 minutos)
  dur_seg = 2 * 60  # 120 s

  mask1 = (t >= 0) & (t < dur_seg)          # 0–120 s
  mask2 = (t >= dur_seg) & (t < 2*dur_seg)  # 120–240 s

  t1 = t[mask1]
  ecg1 = ecg_filtrado[mask1]

  t2 = t[mask2]
  ecg2 = ecg_filtrado[mask2]

  print("Seg1 duración [s]:", t1[-1] - t1[0])
  print("Seg2 duración [s]:", t2[-1] - t2[0])

  plt.figure(figsize=(12,4))
  plt.plot(t1, ecg1)
  plt.title("Segmento 1 (0–120 s)")
  plt.xlabel("Tiempo [s]")
  plt.ylabel("Voltaje [V]")
  plt.grid(True)
  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(12,4))
  plt.plot(t2, ecg2)
  plt.title("Segmento 2 (120–240 s)")
  plt.xlabel("Tiempo [s]")
  plt.ylabel("Voltaje [V]")
  plt.grid(True)
  plt.tight_layout()
  plt.show()
  ```

  + *Seg1 duración [s]: 119.9995*
  + *Seg2 duración [s]: 119.99950000000001*
  

  <img width="950" height="306" alt="image" src="https://github.com/user-attachments/assets/d841ea8e-12a7-4ae3-b602-8f90e6a2aee0" />

  <img width="948" height="306" alt="image" src="https://github.com/user-attachments/assets/63aa3d69-0103-4063-862c-1fccf6897bff" />

  + **Identificar los picos R en cada uno de los segmentos, calcular los intervalos R-R y obtener una nueva señal con dicha información**

    ```python
    from scipy.signal import find_peaks
    import numpy as np

    fs = 2000
    min_rr = 0.3
    dist = int(min_rr * fs)

    # AJUSTE DE PROMINENCIA EN SEGMENTO 1
    print("Probando prominencias para SEGMENTO 1…")

    prom_candidates = [
    np.std(ecg1) * 1.0,
    np.std(ecg1) * 1.5,
    np.std(ecg1) * 2.0,
    np.std(ecg1) * 2.5
    ]

    for prom in prom_candidates:
    pk, _ = find_peaks(ecg1, distance=dist, prominence=prom)
    print(f"prom={prom:.4f} → {len(pk)} picos")

    
    prom1 = np.std(ecg1) * 1.57   

    peaks1, _ = find_peaks(ecg1, distance=dist, prominence=prom1)


    # SEGMENTO 2
    
    prom2 = np.std(ecg2) * 3
    peaks2, _ = find_peaks(ecg2, distance=dist, prominence=prom2)

    print("\nPicos finales en SEGMENTO 1:", len(peaks1))
    print("Picos finales en SEGMENTO 2:", len(peaks2))
    ```

+ *SEGMENTO 1…*
+ *prom=0.0785 → 282 picos*
+ *prom=0.1177 → 257 picos*
+ *prom=0.1570 → 195 picos*
+ *prom=0.1962 → 108 picos*

+ *Picos finales en SEGMENTO 1: 255*
+ *Picos finales en SEGMENTO 2: 237*

```python
# Cálculo de intervalos R-R y nueva señal RR

# Tiempos de cada pico R en cada segmento
tR1 = t1[peaks1]       # tiempos de peaks del seg1
tR2 = t2[peaks2]       # tiempos de peaks del seg2

# Cálculo de intervalos R-R
RR1 = np.diff(tR1)     # señal RR del segmento 1
RR2 = np.diff(tR2)     # señal RR del segmento 2

# Tiempo asociado a cada intervalo R-R
tRR1 = tR1[1:]         # empieza en el segundo pico
tRR2 = tR2[1:]

print("Segmento 1:")
print("  RR1 =", RR1[:10], " ...")
print("  Número de intervalos:", len(RR1))

print("\nSegmento 2:")
print("  RR2 =", RR2[:10], " ...")
print("  Número de intervalos:", len(RR2))
```
+ **Segmento 1:**
  RR1 = [1.1595 0.326  0.327  0.5105 0.355  0.3625 0.435  0.324  0.4655 0.5095]  ...
  Número de intervalos: 254

+ **Segmento 2:**
  RR2 = [0.7655 0.508  0.314  0.47   0.6    0.404  0.6035 0.404  0.6975 0.464 ]  ...
  Número de intervalos: 236

  + **Gráfica de la nueva señal**
```python
# Gráfica de la nueva señal R-R

plt.figure(figsize=(10,4))
plt.plot(tRR1, RR1, '.-', label='RR Segmento 1')
plt.xlabel("Tiempo [s]")
plt.ylabel("Intervalo R-R [s]")
plt.title("Nueva señal de intervalos R-R (Segmento 1)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(tRR2, RR2, '.-', label='RR Segmento 2')
plt.xlabel("Tiempo [s]")
plt.ylabel("Intervalo R-R [s]")
plt.title("Nueva señal de intervalos R-R (Segmento 2)")
plt.grid(True)
plt.legend()
plt.show()
```

<img width="677" height="317" alt="image" src="https://github.com/user-attachments/assets/1dd31dc4-5625-4f76-bca1-6ec08bfd147c" />


<img width="677" height="310" alt="image" src="https://github.com/user-attachments/assets/0f00c805-ed97-4fc2-b543-144c96f93880" />

+ **Comparar los valores de los parámetros básicos de la HRV en el dominio del tiempo, como la media de los intervalos R-R y su desviación estándar, entre ambos segmentos de señal ECG**

```python
# Estadísticas de intervalos RR

import numpy as np

def stats_RR(RR):
    if len(RR) == 0:
        return np.nan, np.nan, np.nan
    RR_mean = np.mean(RR)
    RR_sd   = np.std(RR, ddof=1)
    HR      = 60 / RR_mean
    return RR_mean, RR_sd, HR

RR1_mean, RR1_sd, HR1 = stats_RR(RR1)
RR2_mean, RR2_sd, HR2 = stats_RR(RR2)

print("SEGMENTO 1 (Reposo)")
print("RR medio:", RR1_mean)
print("SDNN:", RR1_sd)  #Representa cuánta variabilidad hay en los intervalos R–R.
print("FC media (lpm):", HR1)

print("\nSEGMENTO 2 (Lectura)")
print("RR medio:", RR2_mean)
print("SDNN:", RR2_sd)
print("FC media (lpm):", HR2)
```
| Parámetro        | Segmento 1 (Reposo)     | Segmento 2 (Lectura)     |
|------------------|--------------------------|----------------------------|
| RR medio (s)     | 0.47212007874015743      | 0.506375                   |
| SDNN (s)         | 0.14237386189807577      | 0.16822658937379914        |
| FC media (lpm)   | 127.0863127874348        | 118.48926191063934         |


El siguiente código organiza los valores calculados de HRV en un DataFrame para mostrar una tabla comparativa entre los segmentos de reposo y lectura.

```python
import pandas as pd

# valores calculados
data = {
    "Parámetro": ["RR medio (s)", "SDNN (s)", "FC media (lpm)"],
    "Segmento 1 (Reposo)": [0.472, 0.142, 127.1],
    "Segmento 2 (Lectura)": [0.506, 0.168, 118.5]
}

# Crear tabla
tabla_hfv = pd.DataFrame(data)

# Mostrar tabla
tabla_hfv
```

| Parámetro        | Segmento 1 (Reposo) | Segmento 2 (Lectura) |
|------------------|---------------------|------------------------|
| RR medio (s)     | 0.472               | 0.506                  |
| SDNN (s)         | 0.142               | 0.168                  |
| FC media (lpm)   | 127.100             | 118.500                |

El análisis comparativo de los parámetros básicos de la variabilidad de la frecuencia cardíaca (HRV) en el dominio del tiempo permitió identificar diferencias significativas entre los dos segmentos de la señal ECG evaluados. En el primer segmento, correspondiente a la condición de reposo, se obtuvo un intervalo R–R medio menor y una desviación estándar (SDNN) relativamente reducida, en conjunto con una frecuencia cardíaca más elevada. Este comportamiento es indicativo de una menor variabilidad del ritmo cardiaco y sugiere una mayor influencia del sistema nervioso simpático, asociado a estados de activación fisiológica o demanda metabólica incrementada.


Por otro lado, en el segundo segmento —registrado durante la lectura— se observó un aumento tanto en el intervalo R–R medio como en la desviación estándar de los intervalos, acompañado de una disminución en la frecuencia cardíaca. Este patrón refleja un incremento en la variabilidad cardiaca y es coherente con una mayor participación del sistema nervioso parasimpático, el cual tiende a disminuir la frecuencia cardiaca y a favorecer una regulación más flexible del ritmo autonómico.


Los cambios observados entre ambos segmentos evidencian una modificación clara en el balance autonómico. Mientras que el estado de reposo presentó una predominancia simpática relativa, la actividad de lectura mostró un predominio parasimpático y un aumento de la HRV. Estas diferencias reflejan la capacidad del sistema cardiovascular para adaptarse dinámicamente a distintas condiciones fisiológicas y cognitivas.

# PARTE C

<img width="358" height="627" alt="image" src="https://github.com/user-attachments/assets/b4026862-d4fc-4cfb-9c90-465f7c3b514e" />

+ **Construcción del diagrama de Poincaré**

```python
import matplotlib.pyplot as plt

#  DIAGRAMA DE POINCARÉ

def poincare_plot(RR, title):
    if len(RR) < 2:
        print("No se puede generar el diagrama, muy pocos RR.")
        return
    RR_n = RR[:-1]
    RR_n1 = RR[1:]

    plt.figure(figsize=(6,6))
    plt.scatter(RR_n, RR_n1, s=10, alpha=0.6)
    plt.xlabel("RR[n] (s)")
    plt.ylabel("RR[n+1] (s)")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# Segmento 1 (reposo)
poincare_plot(RR1, "Diagrama de Poincaré – Segmento 1 (Reposo)")

# Segmento 2 (lectura)
poincare_plot(RR2, "Diagrama de Poincaré – Segmento 2 (Lectura)")
```

<img width="472" height="469" alt="image" src="https://github.com/user-attachments/assets/517fc80f-52d3-4a20-98e3-8e19be0c43f5" />



<img width="466" height="479" alt="image" src="https://github.com/user-attachments/assets/b1312bdf-fe6c-4621-97a5-155a867790d7" />

+ **Calcular los valores de los índices tanto de actividad vagal (CVI) como de actividad simpática (CSI) que se obtienen a partir del diagrama de Poincaré**

```python
import numpy as np
import math

def compute_sd1_sd2(RR):
    # Diferencias sucesivas
    dRR = np.diff(RR)

    # SD1
    SD1 = np.sqrt(0.5) * np.std(dRR, ddof=1)

    # SD2
    SD_RR = np.std(RR, ddof=1)
    SD2 = np.sqrt(2*(SD_RR**2) - SD1**2)

    return SD1, SD2

def compute_cvi_csi(SD1, SD2):
    CVI = np.log10(SD1 * SD2)
    CSI = SD2 / SD1
    return CVI, CSI

# Segmento 1
SD1_1, SD2_1 = compute_sd1_sd2(RR1)
CVI_1, CSI_1 = compute_cvi_csi(SD1_1, SD2_1)

# Segmento 2
SD1_2, SD2_2 = compute_sd1_sd2(RR2)
CVI_2, CSI_2 = compute_cvi_csi(SD1_2, SD2_2)


data = {
    "Parámetro": ["SD1 (s)", "SD2 (s)", "CVI", "CSI"],
    "Segmento 1 (Reposo)": [0.1374299599317766, 0.14715157573772465, -1.6941531318834555, 1.0707400832043537],
    "Segmento 2 (Lectura)": [0.16488999533328808, 0.17149828284102417, -1.548545924495526, 1.040076931675548]
}

tabla_indices = pd.DataFrame(data)
tabla_indices
```

| Parámetro | Segmento 1 (Reposo) | Segmento 2 (Lectura) |
|-----------|----------------------|------------------------|
| SD1 (s)   | 0.137430             | 0.164890               |
| SD2 (s)   | 0.147152             | 0.171498               |
| CVI       | -1.694153            | -1.548546              |
| CSI       | 1.070740             | 1.040077               |


## REFERENCIAS BIBLIOGRÁFICAS

Sistema nervioso parasimpático. (2023, October 30). Kenhub. https://www.kenhub.com/es/library/anatomia-es/sistema-nervioso-parasimpatico

LeBouef, T., Yaker, Z., & Whited, L. (2023, May 1). Physiology, autonomic nervous system. StatPearls - NCBI Bookshelf. https://www.ncbi.nlm.nih.gov/books/NBK538516/

Coon, E. (2025, May 9). Generalidades sobre el sistema nervioso autónomo. Manual MSD Versión Para Profesionales. https://www.msdmanuals.com/es/professional/trastornos-neurol%C3%B3gicos/sistema-nervioso-aut%C3%B3nomo/generalidades-sobre-el-sistema-nervioso-aut%C3%B3nomo

Kasahara, Y., Yoshida, C., Saito, M., & Kimura, Y. (2021). Assessments of heart rate and sympathetic and parasympathetic nervous activities of normal mouse fetuses at different stages of fetal development using fetal electrocardiography. Frontiers in Physiology, 12. https://doi.org/10.3389/fphys.2021.652828

Olshansky, B., Sabbah, H. N., Hauptman, P. J., & Colucci, W. S. (2008). Parasympathetic nervous system and heart failure. Circulation, 118(8), 863–871. https://doi.org/10.1161/circulationaha.107.760405

Electrophysiology, T. F. O. T. E. S. O. C. T. N. A. (1996). Heart rate variability. Circulation, 93(5), 1043–1065. https://doi.org/10.1161/01.cir.93.5.1043

Pichot, V., Roche, F., Celle, S., Barthélémy, J., & Chouchou, F. (2016). HRVAnalysis: a free software for analyzing cardiac autonomic activity. Frontiers in Physiology, 7, 557. https://doi.org/10.3389/fphys.2016.00557

Wang, B., Liu, D., Gao, X., & Luo, Y. (2022). Three‐Dimensional poincaré plot analysis for heart rate variability. Complexity, 2022(1). https://doi.org/10.1155/2022/3880047




    






