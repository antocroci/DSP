#  Estimación de Dirección de Arribo (DOA) usando TDOA

Este repositorio contiene el desarrollo completo de un trabajo de investigación centrado en la estimación de la dirección de arribo (DOA) de fuentes sonoras mediante técnicas basadas en la diferencia de tiempos de arribo (TDOA). El objetivo principal es analizar el comportamiento y la precisión de distintos algoritmos en escenarios simulados, como paso preliminar para su aplicación en entornos reales de localización acústica.

Para las simulaciones se utilizó la librería pyroomacoustics, que permite recrear condiciones acústicas variadas, incluyendo ambientes anecoicos y reverberantes. El estudio evalúa el impacto de variables clave como el ángulo azimutal de la fuente, la distancia entre la fuente y el array, el nivel de ruido, el tipo de ruido y la posicion del arreglo. 

Este trabajo busca sentar las bases para el desarrollo de sistemas de localización acústica multicanal, con aplicaciones en campos como la robótica, los audífonos inteligentes, la realidad aumentada y los sistemas de vigilancia sonora.

---

## ¿Qué hace este código?

* Crea una sala simulada con un arreglo lineal de micrófonos.
* Genera una fuente sonora.
* Calcula los TDOA entre pares de micrófonos.
* Estima el ángulo de llegada (DOA).
* Compara el ángulo real vs. el estimado y lo grafica.

---

## Estructura básica
```
TP_DOA/
│
├── datos/                 # Señales simuladas, configuraciones, logs
├── simulacion.py
├── tdoa.py
├── doa.py
├── evaluacion.py
├── main.py
├── informe/
│   ├── informe.tex/.docx
│   └── figuras/
└── README.md              # Instrucciones para correr el código
```
