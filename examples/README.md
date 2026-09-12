# Archivos de ejemplo del procesamiento por lotes

Entrada y salida reales de la pestaña **📂 Procesamiento por lotes** de la demo, para poder
probarla sin tener que fabricarse un archivo antes.

| Archivo | Qué es |
|---|---|
| `aspirantes_ejemplo.csv` | 10 aspirantes de entrada. Es el archivo que ofrece la demo en «⬇️ Descargar archivo de ejemplo» |
| `predicciones_ejemplo.csv` | la salida que produce ese archivo con el modelo de servicio |

Están aquí, y no en `data/`, porque `data/**` está en `.gitignore` y estos archivos tienen
que viajar al repositorio: la demo los sirve como descarga.

`predicciones_ejemplo.csv` se genera **en las mismas condiciones que el despliegue**, es
decir, sin el informe de métricas del training pipeline delante: ese informe tampoco se
versiona, así que en Streamlit Cloud `MAE_MODELO` toma el valor medido de respaldo. De ahí
salen los límites del intervalo, que son `predicción ± MAE`; si se regenera el archivo en
local con el informe presente, esas dos columnas cambian en la quinta cifra decimal.

## Cómo usarlos

1. Abre la demo (<https://admisiones-project-cd.streamlit.app/>) y ve a la pestaña
   **📂 Procesamiento por lotes**.
2. Descarga `aspirantes_ejemplo.csv` desde la propia demo, o toma este de aquí.
3. Súbelo con **Sube tu archivo**.
4. Revisa el resumen por cesta y la tabla, y descarga las predicciones. El CSV descargado es
   equivalente a `predicciones_ejemplo.csv`.

Lo mismo desde la línea de comandos, con el script que hay detrás de la pestaña:

```bash
uv run python src/pipelines/inference_pipeline/inference_pipeline.py \
    --modelo models/modelo_produccion.joblib \
    --datos examples/aspirantes_ejemplo.csv \
    --salida examples/predicciones_ejemplo.csv --mostrar 10
```

## Qué demuestra cada fila

El archivo de entrada no es una muestra al azar: cada fila está puesta para enseñar un
comportamiento distinto de la demo.

| `id` | Para qué está |
|---|---|
| `A-001`, `A-008` | perfiles fuertes: salen como opción **segura** |
| `A-002`, `A-003`, `A-009`, `A-010` | perfiles medios: **probable** |
| `A-004`, `A-007` | perfiles flojos: **ambiciosa** |
| `A-005` | cae en el tramo bajo: la columna `advertencia` se marca y la demo avisa |
| `A-006` | **sin puntaje TOEFL** (`n/a`): el modelo imputa el dato que falta |
| `A-007` | **sin cartas de recomendación** (`n/a`): mismo caso, en otra columna |
| `A-009` | **celda vacía** en TOEFL y `n/a` en el rating: las dos formas de decir «no lo sé» |

Las columnas `id` y `nombre` no entran al modelo, pero **vuelven a salir** en el archivo de
predicciones: son las que permiten saber de quién es cada número.

## Formato de entrada

Se aceptan `.csv` y `.parquet`, con los nombres del dataset original (`GRE Score`, `LOR `) o
en minúsculas con guion bajo (`gre_score`, `lor`). Las celdas vacías y los `n/a` valen; cada
fila necesita al menos 4 de los 7 datos. Un valor fuera del dominio documentado —un GRE de
900, un `University Rating` de 9— **rechaza el archivo entero** con un mensaje que dice qué
regla se incumplió y en qué columna, en vez de generar una predicción que nadie debería
usar.

## Columnas de salida

A las columnas de entrada se les añaden las cinco del modelo:

| Columna | Qué es |
|---|---|
| `prediccion` | probabilidad estimada de admisión, entre 0 y 1 |
| `cesta` | `segura`, `probable` o `ambiciosa` |
| `limite_inferior`, `limite_superior` | rango de referencia de ±1 MAE alrededor de la predicción |
| `advertencia` | `True` si la predicción cae en el tramo con menos ejemplos y más error |
