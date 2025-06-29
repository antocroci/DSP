#  Estimación de Dirección de Arribo (DOA) usando TDOA

Este proyecto simula cómo un arreglo de micrófonos puede estimar de dónde viene un sonido usando **retardos temporales (TDOA)**. Probamos diferentes algoritmos de correlación como **CCC, GCC, PHAT y SCOT** en distintos entornos simulados (con y sin reverberación), usando la librería `pyroomacoustics`.

---

## ¿Qué hace este código?

* Crea una sala simulada con un arreglo lineal de micrófonos.
* Genera una fuente sonora.
* Calcula los TDOA entre pares de micrófonos.
* Estima el ángulo de llegada (DOA).
* Compara el ángulo real vs. el estimado y lo grafica.

---

## Estructura básica
## cambiar!!!
```
src/
├── doa/           # estimación de TDOA y DOA
├── sim/           # setup de sala, micrófonos y fuente
├── utils/         # helpers para gráficos, etc.
notebooks/
├── demo.ipynb     # ejemplo completo paso a paso
results/           # figuras y resultados generados
```
