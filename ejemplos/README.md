# Archivos de ejemplo de la demo

Entrada y salida reales del **procesamiento por lotes**, para poder probar la demo sin
tener que fabricarse un archivo antes.

| Archivo | Qué es |
|---|---|
| `aspirantes_ejemplo.csv` | 8 aspirantes de entrada. Es el archivo que ofrece la demo en «⬇️ Descargar archivo de ejemplo» |
| `predicciones_ejemplo.csv` | la salida que produce ese archivo con el modelo publicado |

## Cómo usarlos

1. Abre la demo (<https://admisiones-project-cd.streamlit.app/>) y ve a la pestaña
   **📂 Procesamiento por lotes**.
2. Descarga `aspirantes_ejemplo.csv` desde la propia demo, o toma este de aquí.
3. Súbelo con **Sube tu archivo**.
4. Revisa la tabla y descarga las predicciones. El resultado debe coincidir con
   `predicciones_ejemplo.csv`.

Lo mismo, desde la línea de comandos:

```bash
uv run python src/pipelines/inference_pipeline/inference_pipeline.py \
    --modelo data/06_models/modelo_final_automl.joblib \
    --datos ejemplos/aspirantes_ejemplo.csv \
    --salida ejemplos/predicciones_ejemplo.csv --mostrar 8
```

## Qué demuestra cada fila

El archivo de entrada no es una muestra al azar: cada fila está puesta para enseñar un
comportamiento distinto de la demo.

| `id` | Para qué está |
|---|---|
| `A-001`, `A-008` | perfiles fuertes: salen como opción **segura** |
| `A-002`, `A-003` | perfiles medios: **probable** |
| `A-004`, `A-007` | perfiles flojos: **ambiciosa** |
| `A-005` | cae bajo el umbral de 0.55: la columna `advertencia` se marca y la demo avisa |
| `A-006` | **sin puntaje TOEFL** (`n/a`): el modelo imputa el dato que falta |
| `A-007` | **sin cartas de recomendación**: mismo caso, en otra columna |

Las columnas `id` y `nombre` no entran al modelo, pero **vuelven a salir** en el archivo de
predicciones: son las que permiten saber de quién es cada número.

## Formato de entrada

Se aceptan `.csv` y `.parquet`, con los nombres del dataset original (`GRE Score`, `LOR `) o
en minúsculas con guion bajo (`gre_score`, `lor`). Las celdas vacías y los `n/a` valen; cada
fila necesita al menos 4 de los 7 datos. Un valor fuera del dominio documentado —un GRE de
900, un `University Rating` de 9— **rechaza el archivo entero** con un mensaje que dice qué
fila y qué columna, en vez de generar una predicción que nadie debería usar.
