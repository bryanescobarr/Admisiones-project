# Template for data science with Python 3.12 and devcontainer

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/JoseRZapata/data-science-project-template/branch/main/graph/badge.svg?token=G9K6YJ8J6W)](https://codecov.io/gh/JoseRZapata/data-science-project-template)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

**Demo en línea del modelo: <https://admisiones-project-cd.streamlit.app/>**

This is a data science project template created with [Cookiecutter] to help you start your next data science or machine learning project quickly and efficiently. It includes a well-organized folder structure, essential tools for code quality, testing, and documentation, and follows best practices in the industry.

Using the data science project template <https://github.com/JoseRZapata/data-science-project-template>

- `Python` = `3.12`
- `devcontainer` to work in `VSCode` or [GitHub Codespaces](https://github.com/features/codespaces) using the same environment as in production.

## 🎓 Demo: predicción de admisión a posgrado

**🔗 Demo en línea: <https://admisiones-project-cd.streamlit.app/>**

[![Abrir la demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://admisiones-project-cd.streamlit.app/)

La demo tiene **dos pestañas**, sobre el mismo modelo:

| Pestaña | Qué hace |
|---|---|
| 🎯 **Predicción individual** | un aspirante introduce su perfil y obtiene su probabilidad, su cesta (**segura**, **probable** o **ambiciosa**) y el desglose de cuánto aportó cada dato |
| 📂 **Procesamiento por lotes** | se sube un archivo con varios aspirantes y se obtienen todas las predicciones juntas, con resumen por cesta, tabla y descarga en CSV |

### Cómo usar el procesamiento por lotes

1. Abre la pestaña **📂 Procesamiento por lotes**.
2. Pulsa **⬇️ Descargar archivo de ejemplo** (o toma `examples/aspirantes_ejemplo.csv`) para
   ver el formato esperado.
3. Sube tu archivo `.csv` o `.parquet` con una fila por aspirante.
4. Revisa el resumen —cuántos aspirantes, probabilidad media y cuántos con advertencia—, el
   reparto por cesta y la tabla de resultados.
5. Pulsa **⬇️ Descargar predicciones (CSV)**.

**Qué debe tener el archivo:** las siete columnas del perfil (`GRE Score`, `TOEFL Score`,
`University Rating`, `SOP`, `LOR`, `CGPA`, `Research`), con los nombres del dataset original
o en minúsculas con guion bajo. Puedes **dejar celdas vacías o escribir `n/a`** —el modelo
imputa, igual que el «No lo sé» del formulario— siempre que cada fila tenga al menos 4 de
los 7 datos. Las **columnas de más se conservan**: si incluyes un `id` o el nombre del
aspirante, vuelven en el archivo de resultados.

A la salida se añaden cinco columnas: `prediccion`, `cesta`, `limite_inferior`,
`limite_superior` y `advertencia`.

Un valor fuera del dominio documentado —un GRE de 900, un rating de 9— **rechaza el archivo
entero**, con la lista de reglas incumplidas y sin tumbar la aplicación:

```text
No se pudo procesar el archivo. 1 regla(s) incumplida(s):
- [gre_score] fuera_de_rango: valores fuera de [0, 340]: [900] (1 fila)
```

**Dónde vive la lógica.** La pestaña no valida, no predice y no clasifica: encadena las
funciones de `src/pipelines/inference_pipeline/inference_pipeline.py` —`leer_datos_nuevos`,
`separar_contexto`, `preparar_datos_nuevos`, `predecir_lote` y `resumir_lote`—, las mismas
que ejecuta el script de línea de comandos. El archivo subido se vuelca a un temporal porque
`leer_datos_nuevos()` recibe una ruta, y **no se escribe nada en `data/`**: el disco de
Streamlit Cloud es efímero y la salida se entrega por descarga.

Para lotes grandes o automatizados, el mismo trabajo desde la terminal:

```bash
uv run python src/pipelines/inference_pipeline/inference_pipeline.py \
    --modelo models/modelo_produccion.joblib \
    --datos examples/aspirantes_ejemplo.csv \
    --salida predicciones.csv --mostrar 10
```

### Evidencia de funcionamiento

En `examples/` están la entrada y la salida reales del lote, versionadas y generadas con el
modelo de servicio:

| Archivo | Qué es |
|---|---|
| `examples/aspirantes_ejemplo.csv` | 10 aspirantes, tres de ellos con datos incompletos |
| `examples/predicciones_ejemplo.csv` | lo que produce la demo con ese archivo |
| `examples/README.md` | qué demuestra cada fila y cómo reproducirlo |

Resultado de ese lote: **10 aspirantes procesados**, probabilidad media **75 %**, reparto
2 `segura` / 5 `probable` / 3 `ambiciosa` y **1 predicción marcada con advertencia** (el
aspirante `A-005`). Las tres filas con datos ausentes —`A-006` sin TOEFL, `A-007` sin cartas
y `A-009` con la celda vacía— se procesan igual, porque el pipeline del modelo imputa con la
mediana aprendida en entrenamiento.

Quince pruebas automáticas ejercitan la aplicación real con `streamlit.testing.v1.AppTest`,
incluida una que **sube el archivo de ejemplo y comprueba que lo que se descargaría es
idéntico a `examples/predicciones_ejemplo.csv`**, y tres que suben archivos inválidos y
verifican que la app sigue en pie:

```bash
uv run pytest tests/test_app.py -q   # 18 passed
```

### Ejecutar en local

```bash
uv sync --all-extras --dev
uv run streamlit run app.py
```

Se abre en <http://localhost:8501>, con las dos pestañas. No hace falta ejecutar ningún
notebook ni ningún pipeline antes: el modelo que sirve la demo ya está versionado en
`models/modelo_produccion.joblib`.

Para verificar que todo funciona sin abrir el navegador:

```bash
uv run pytest tests/test_app.py tests/test_prediccion.py -v
```

### Qué modelo sirve la demo

`models/modelo_produccion.joblib`, el artefacto que produce el **training
pipeline**. Antes se servía el `.joblib` que se exportó a mano desde el notebook
`06.Seleccion-de-modelo-AutoML`; ahora la demo sirve lo que genera la cadena reproducible
—feature pipeline, chequeos de partición, entrenamiento y validación—, no una pieza suelta.
Los artefactos del POC siguen versionados como referencia histórica.

| | POC (`modelo_final_automl`) | Servicio (`modelo_produccion`) |
|---|---|---|
| MAE en prueba | 0.0476 | **0.0467** |
| Acierto de cesta | 80.5 % | **84.7 %** |
| Predicción mínima | 0.447 | 0.4066 |
| Cómo se genera | a mano, desde un notebook | un comando, reproducible |
| Chequeos de partición y diagnóstico de generalización | no | sí |

#### Cómo regenerarlo

```bash
uv run python src/pipelines/feature_pipeline/feature_pipeline.py
uv run python src/pipelines/training_pipeline/train_pipeline.py \
    --modelo-salida models/modelo_produccion.joblib
git add models/modelo_produccion.joblib
```

El artefacto vive en `models/` —la carpeta que la estructura del proyecto reserva para los
modelos finales— y no bajo `data/`, que está entero en `.gitignore`. Así basta un `git add`
normal, sin `-f`, y **Streamlit Cloud puede servirlo**, porque solo sirve lo que está
versionado. El artefacto ocupa unos 4.2 MiB.

Si se regenera con otro entorno, hay que **actualizar `requirements.txt` en el mismo
commit**: las versiones fijadas ahí son las que serializaron el artefacto, y un desajuste de
`scikit-learn`, `numpy` o `joblib` impide deserializarlo en el despliegue.

#### De dónde salen las cifras que muestra la demo

`MAE_MODELO` se lee de `data/08_reporting/metricas_training_pipeline.parquet`, el informe
que escribe el propio training pipeline, en la fila del modelo servido. Ese informe **no se
versiona** (la regla es versionar solo el artefacto de servicio), así que en el despliegue
`src/inference/prediccion.py` recurre al valor medido en la ejecución que generó el
`.joblib`. Lo mismo con `PREDICCION_MINIMA`, que además todavía no es una columna del
informe. Una prueba recalcula ambas cifras desde los datos crudos versionados, de modo que
no puedan quedarse describiendo a otro modelo:

```bash
uv run pytest tests/test_prediccion.py -q
```

### Publicar en Streamlit Community Cloud

1. Entrar en <https://share.streamlit.io> con la cuenta de GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository `bryanescobarr/Admisiones-project`, branch `main`, main file path `app.py`,
   y en *Advanced settings* elegir **Python 3.12**.
4. **Deploy**.

El repositorio ya trae lo que necesita el despliegue: `requirements.txt` con las versiones
**fijadas** (el `.joblib` se serializó con `scikit-learn 1.9.0` y `pandas 3.0.5`, y otras
versiones pueden no deserializarlo) y `packages.txt` con `libgomp1`, la librería de OpenMP
que necesita scikit-learn.

### Cómo está construida

| Archivo | Qué hace |
|---|---|
| `app.py` | interfaz: las dos pestañas, resultados, gráfico de aportes, advertencias |
| `src/pipelines/inference_pipeline/inference_pipeline.py` | la inferencia por lotes: **la pestaña y el script de línea de comandos ejecutan el mismo código** |
| `examples/` | archivos de entrada y salida de ejemplo del lote |
| `src/inference/prediccion.py` | lógica: cargar el modelo de servicio, predecir, clasificar en cestas, explicar con SHAP |
| `models/modelo_produccion.joblib` | el artefacto que se sirve, generado por `train_pipeline.py` |
| `notebooks/7-deploy/08.Demo-de-la-aplicacion-BER-2026-08-21.ipynb` | documentación de la demo y del despliegue |

La lógica vive fuera de la interfaz para poder probarla: una interfaz rota se ve, una regla
de negocio mal escrita no. La demo admite campos vacíos (el pipeline imputa), muestra un
rango de referencia junto a la cifra y **avisa explícitamente** cuando la predicción cae en
el tramo donde el modelo tiende a ser optimista.

## 🏭 Feature pipeline

Script ejecutable que automatiza lo que antes solo existía dentro de los notebooks
`2-exploration` y `4-feat_eng`: leer la fuente cruda, limpiarla, tiparla, validarla y dejar
las features guardadas en disco.

```bash
uv run python src/pipelines/feature_pipeline/feature_pipeline.py
```

```text
data/01_raw/Admission_Predict.csv  ->  data/04_feature/admisiones_features.parquet
```

| Opción | Para qué sirve |
|---|---|
| `--entrada` | CSV de datos crudos (por defecto, el del repositorio) |
| `--salida` | parquet de features a generar |
| `--con-derivados` | añade `indice_academico`, `sop_lor_media` y `rating_x_research` |

Qué hace: normaliza los nombres de columna, unifica las representaciones de nulo (`n/a`,
`-`, celda vacía), elimina duplicados exactos, asigna los tipos nullable de pandas, **valida
el dominio documentado** en `data/01_raw/Informacion.txt` y recorta los valores imposibles.

Qué **no** hace, a propósito: imputar y escalar. Esos pasos aprenden sus parámetros de los
datos (la mediana, la media, la desviación) y, calculados antes de particionar, filtrarían
información del conjunto de prueba al de entrenamiento. Viven en el `Pipeline` de
scikit-learn de `4-feat_eng`, que se ajusta solo con *train*.

### Validación de datos e integridad

Las features **no se guardan si los datos no pasan el contrato**. Hay tres puertas de
calidad, todas antes de escribir el parquet, así que un fallo deja intacto el archivo
anterior en vez de sustituirlo por datos corruptos:

| Puerta | Qué comprueba |
|---|---|
| `ESQUEMA_FUENTE` | la fuente ya tipada: tipos, rangos de `Informacion.txt`, categorías válidas, % de nulos, registros duplicados e integridad entre campos |
| `ESQUEMA_FEATURES` | lo que se va a persistir: el dominio otra vez y la coherencia de los atributos derivados con su fórmula |
| `comparar_datasets` | integridad entre la entrada y la salida: ni filas inventadas, ni columnas perdidas, ni valores o nulos que no estaban en el origen |

El motor que aplica los contratos está en `src/data/validacion.py`; los contratos, con las
constantes que los justifican, en el propio `feature_pipeline.py`. Se recogen **todas** las
violaciones antes de fallar, y el script termina con código de salida 1 y un informe
legible:

```console
$ uv run python src/pipelines/feature_pipeline/feature_pipeline.py --entrada datos_malos.csv
ERROR Validacion fallida en la fuente tipada: 2 regla(s) incumplida(s)
  - [gre_score] fuera_de_rango: valores fuera de [0, 340]: [900] (1 fila)
  - [chance_of_admit] exceso_de_nulos: 33.3% de nulos, maximo admitido 0.0% (1 fila)
ERROR No se persistieron features: data/04_feature/admisiones_features.parquet no se ha modificado
```

## 🏋️ Training pipeline

Segunda etapa FTI: lee las features del paso anterior, parte train/test, entrena, evalúa y
guarda el modelo con sus métricas.

```bash
uv run python src/pipelines/training_pipeline/train_pipeline.py
```

```text
data/04_feature/admisiones_features.parquet
  -> data/06_models/modelo_training_pipeline.joblib
  -> data/08_reporting/metricas_training_pipeline.parquet
```

| Opción | Para qué sirve |
|---|---|
| `--modelo` | `dummy`, `heuristica`, `ridge`, `extra_trees` (por defecto) o `automl` |
| `--presupuesto` | segundos de búsqueda, solo con `--modelo automl` |
| `--proporcion-test` / `--semilla` | partición (0.25 y 42, como en los notebooks) |
| `--sin-referencias` | mide solo el modelo elegido, sin el dummy ni la heurística |
| `--particion-estricta` | detiene el entrenamiento también ante una advertencia de deriva |
| `--chequeos-particion` | ruta del informe de chequeos de la partición |
| `--features` / `--modelo-salida` / `--metricas` | rutas de entrada y salida |

Salida de una ejecución sobre los datos del repositorio:

```text
     modelo    MAE   RMSE      R2   MAPE  spearman  MAE_cv  MAE_entrenamiento  brecha_train_cv  mejora_vs_dummy_%
extra_trees 0.0467 0.0659  0.7977 0.0734    0.8930  0.0431                0.0           0.0431            60.8181
 heuristica 0.0643 0.0845  0.6667 0.1037    0.8292     NaN                NaN              NaN            46.0545
      dummy 0.1191 0.1465 -0.0017 0.1799       NaN     NaN                NaN              NaN             0.0000
```

Tres detalles que no son decorativos:

- **La partición ocurre antes de ajustar nada** y el preprocesamiento vive dentro del
  `Pipeline`, así que la mediana de imputación y la media del escalado se aprenden solo con
  *train*. Por eso el feature pipeline deja esos pasos sin hacer.
- **Cada ejecución mide también las referencias** (`dummy` y `heuristica`): un MAE de 0.047
  no significa nada solo; comparado con 0.119 y 0.064, sí.
- **El artefacto se guarda aparte** del modelo que sirve la demo
  (`modelo_final_automl.joblib`): entrenar no debe cambiar lo que ven los usuarios sin que
  alguien lo decida.

La receta de preprocesamiento está en `src/data/preprocesamiento.py`, y una prueba
comprueba que reproduce exactamente el pipeline ajustado en el notebook `4-feat_eng`.

### Chequeos de la partición train/test

Entre partir y entrenar hay una puerta: `validar_particion_train_test()`
(`src/data/particion.py`) comprueba que el modelo no se va a examinar con datos que ya vio y
que los dos conjuntos representan el mismo problema. Los umbrales son los de la suite
`train_test_validation` de deepchecks.

| Chequeo | Qué mide | Umbral | Severidad si falla |
|---|---|---|---|
| `indices_solapados` | filas presentes en los dos conjuntos | 0 | **error** |
| `filas_compartidas` | perfiles de prueba que ya estaban en entrenamiento | 5 % | **error** (advertencia por debajo) |
| `proporcion_de_la_particion` | desviación de la proporción pedida | 5 puntos | advertencia |
| `tamano_del_conjunto_de_prueba` | ratio test/train y mínimo de filas | 0.01 / 30 filas | advertencia |
| `deriva_del_objetivo` | Kolmogorov-Smirnov sobre `chance_of_admit` | D ≤ 0.2 | advertencia |
| `deriva_de_la_variable` | KS por predictor | D ≤ 0.2 | advertencia |
| `categorias_nuevas_en_prueba` | valores que el modelo nunca vio | 0 | advertencia |
| `deriva_de_los_nulos` | reparto de datos ausentes | 10 puntos | advertencia |
| `deriva_multivariante` | AUC de un clasificador que intenta distinguir train de test | ≤ 0.65 | advertencia |

**La fuga detiene el pipeline; la deriva se avisa.** Entrenar con fuga es peor que no
entrenar, porque produce métricas excelentes en las que alguien va a confiar. La deriva, en
cambio, aparece por azar en una partición aleatoria de 471 filas, así que se registra y se
sigue; con `--particion-estricta` cualquier advertencia también detiene el proceso.

El informe se guarda **siempre**, pasen o no los chequeos, en
`data/08_reporting/chequeos_particion.parquet`:

```text
                      chequeo           columna severidad  estadistico  umbral
            indices_solapados                          ok          NaN     NaN
            filas_compartidas                          ok       0.0000     NaN
   proporcion_de_la_particion                          ok       0.0005    0.05
          deriva_del_objetivo   chance_of_admit        ok       0.0615    0.20
        deriva_de_la_variable              cgpa        ok       0.0689    0.20
         deriva_multivariante                          ok       0.4804    0.65
```

### Validación del modelo

Una métrica en prueba es **un número con una sola muestra**: cambia la semilla y cambia el
número. Cada ejecución mide las mismas métricas en los tres escenarios y traduce la
comparación en veredictos con acciones (`src/model/validacion.py`).

| Opción | Para qué sirve |
|---|---|
| `--validador` | `kfold`, `repetido` (por defecto), `estratificado` o `series_temporales` |
| `--pliegues` / `--repeticiones` | tamaño de la validación cruzada (5 × 5) |
| `--sin-curva` | omite la curva de aprendizaje, que es la parte cara |
| `--grafico` | guarda además la curva como PNG (necesita `matplotlib`) |
| `--reportes` | directorio donde se deja la evidencia |

```text
 metrica  entrenamiento  validacion_cruzada  desviacion_cv  prueba  brecha_train_cv  diferencia_cv_prueba
     MAE            0.0              0.0445         0.0046  0.0467           0.0445                0.0022
    RMSE            0.0              0.0644         0.0072  0.0659           0.0644                0.0014
      R2            1.0              0.7897         0.0444  0.7977          -0.2103                0.0081
    MAPE            0.0              0.0724         0.0091  0.0734           0.0724                0.0011
spearman            1.0              0.8905         0.0318  0.8930          -0.1095                0.0025

Diagnostico [sobreajuste]: sobreajuste | MAE 0.0000 en entrenamiento y 0.0445 en validacion (100% de brecha)
   -> limitar la capacidad del modelo (profundidad, min_samples_leaf), regularizar o conseguir mas datos
Diagnostico [subajuste]: capacidad suficiente | R2 de validacion cruzada 0.7897 (umbral 0.5)
Diagnostico [estabilidad]: estable | desviacion entre pliegues 0.0046, un 10% del MAE medio
Diagnostico [degradacion_cv_prueba]: coherente | 0.0445 en validacion y 0.0467 en prueba (5%, umbral 10%)
```

**Lo que el diagnóstico encontró en el modelo actual:** el `extra_trees` tiene **MAE 0 en
entrenamiento** —los árboles sin podar memorizan— y 0.0445 en validación cruzada. Generaliza
bien (la prueba confirma la validación, 5 % de diferencia), pero la brecha está ahí y el
informe la nombra en vez de esconderla.

| Artefacto en `data/08_reporting/` | Contenido |
|---|---|
| `validacion_modelo.parquet` | métricas en entrenamiento, validación cruzada y prueba, con el validador y la semilla usados |
| `diagnostico_generalizacion.parquet` | veredicto, evidencia y acción recomendada por aspecto |
| `curva_aprendizaje.parquet` / `.png` | error según el número de muestras: ¿ayudarían más datos? |
| `segmentos_debiles.parquet` | error por tercio de cada atributo: ¿a quién le funciona peor? |

La validación es **reproducible**: el validador, los pliegues, las repeticiones y la semilla
se guardan como columnas del propio informe, así que el artefacto dice cómo se produjo.

## 🔮 Inference pipeline

Tercera etapa FTI: carga el modelo entrenado, lee un archivo de aspirantes nuevos, les
aplica las mismas transformaciones del entrenamiento y guarda las predicciones.

```bash
uv run python src/pipelines/inference_pipeline/inference_pipeline.py --datos aspirantes.csv --mostrar 5
```

```text
data/06_models/modelo_training_pipeline.joblib + aspirantes.csv
  -> data/07_model_output/predicciones.parquet
```

| Opción | Para qué sirve |
|---|---|
| `--modelo` | artefacto a cargar (por defecto, el del training pipeline) |
| `--datos` | CSV o parquet de aspirantes; por defecto, el archivo crudo del proyecto |
| `--salida` | archivo de predicciones, `.parquet` o `.csv` |
| `--mostrar N` | escribe las primeras N predicciones en el log |
| `--sin-guardar` | inspecciona el lote sin escribir nada |

**Aquí no se reimplementa ninguna transformación.** El `.joblib` guarda el `Pipeline`
completo con los parámetros aprendidos en entrenamiento —la mediana de cada columna, la
media y la desviación del escalado, el mapeo del encoder—, así que `modelo.predict()`
aplica esa misma secuencia. Lo que sí hace el script antes es la **puesta en formato**:
normaliza los nombres (`'LOR '`), unifica los `'n/a'`, tipa y valida contra el dominio
documentado, **sin exigir la columna objetivo** (un aspirante que aún no fue admitido no la
tiene) y **permitiendo perfiles repetidos** (dos aspirantes distintos pueden coincidir).

Cada fila sale con su cesta, su rango de ±1 MAE y la advertencia del tramo bajo, con la
misma lógica que la demo (`src/inference/prediccion.py`); las columnas que no entran al
modelo —un `id`, la universidad— vuelven a salir en el archivo de predicciones.

```console
$ uv run python src/pipelines/inference_pipeline/inference_pipeline.py --datos aspirantes.csv --mostrar 2
INFO Modelo cargado de data/06_models/modelo_training_pipeline.joblib (4324.2 KB): pasos ['preparacion', 'modelo'], estimador ExtraTreesRegressor
INFO Leidas 2 filas x 8 columnas de aspirantes.csv
INFO Datos preparados: 2 filas, 1 valores ausentes que el modelo imputara
INFO Predicciones generadas: 2 filas, media 0.727, rango [0.599, 0.855], +-0.0476 de MAE
INFO Distribucion por cesta:
    cesta  n  porcentaje  prediccion_media
ambiciosa  1        50.0             0.599
   segura  1        50.0             0.855
 prediccion     cesta  limite_inferior  limite_superior  advertencia
     0.8551    segura           0.8075           0.9027        False
     0.5989 ambiciosa           0.5513           0.6465        False
```

Un dato imposible —un GRE de 900, un rating 9, un perfil casi vacío— detiene el lote con un
informe y código de salida 1, **sin escribir ningún archivo de predicciones**.

## ✨ Features and Tools

Information about all the features and tools used in this project: <https://joserzapata.github.io/data-science-project-template/#features-and-tools>

Features                                     | Package  | Why?
 ---                                         | ---      | ---
Dependencies and env                         | [UV] | [article](https://astral.sh/blog/uv)
Lint - Format, sort imports  (Code Quality)  | [Ruff] | [article](https://www.sicara.fr/blog-technique/boost-code-quality-ruff-linter)
Static type checking                         | [Mypy] | [article](https://python.plainenglish.io/does-python-need-types-79753b88f521)
code security                                | [bandit] | [article](https://blog.bytehackr.in/secure-your-python-code-with-bandit)
Code quality & security each commit          | [pre-commit] | [article](https://dev.to/techishdeep/maximize-your-python-efficiency-with-pre-commit-a-complete-but-concise-guide-39a5)
Test code                                    | [Pytest] | [article](https://realpython.com/pytest-python-testing/)
Test coverage                                | [coverage.py] [codecov] | [article](https://martinxpn.medium.com/test-coverage-in-python-with-pytest-86-100-days-of-python-a3205c77296)
Project Template                             | [Cruft] or [Cookiecutter] | [article](https://medium.com/@bctello8/standardizing-dbt-projects-at-scale-with-cookiecutter-and-cruft-20acc4dc3f74)
Folder structure for data science projects   | [Data structure] | [article](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)
Template for pull requests                   | [Pull Request template] | [article](https://www.awesomecodereviews.com/pull-request-template/)
Template for notebooks                       | [Notebook template] |

## Set up the environment

1. Initialize git in local:

    ```bash
    make init_git
    ```

1. Set up the environment:

    ```bash
    make install_env
    ```

1. Activate virtual environment:

    ```bash
    source .venv/bin/activate
    ```

1. Install libraries for data science and machine learning:

    ```bash
    make install_data_libs
    ```

## Install dependencies

After init the environment to install a new package, run:

```bash
uv add <package-name>
```

Example to install [plotly](https://plotly.com/python/) in dev group:

```bash
uv add --group dev plotly
```

## 🗃️ Project structure

- [Data structure]
- [Pipelines based on Feature/Training/Inference Pipelines](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)

```bash
.
├── codecov.yml                         # configuration for codecov
├── .code_quality
│   ├── mypy.ini                        # mypy configuration
│   └── ruff.toml                       # ruff configuration
├── data
│   ├── 01_raw                          # raw immutable data
│   ├── 02_intermediate                 # typed data
│   ├── 03_primary                      # domain model data
│   ├── 04_feature                      # model features
│   ├── 05_model_input                  # often called 'master tables'
│   ├── 06_models                       # serialized models
│   ├── 07_model_output                 # data generated by model runs
│   ├── 08_reporting                    # reports, results, etc
│   └── README.md                       # description of the data structure
├── docs                                # documentation for your project
├── examples                            # entrada y salida de ejemplo del procesamiento por lotes
├── .editorconfig                       # editor configuration
├── .github                             # github configuration
│   ├── dependabot.md                   # github action to update dependencies
│   ├── pull_request_template.md        # template for pull requests
│   └── workflows                       # github actions workflows
│       ├── ci.yml                      # run continuous integration (tests, pre-commit, etc.)
│       ├── dependency_review.yml       # review dependencies
│       ├── docs.yml                    # build documentation (mkdocs)
│       └── pre-commit_autoupdate.yml   # update pre-commit hooks
├── .gitignore                          # files to ignore in git
├── Makefile                            # useful commands to setup environment, run tests, etc.
├── models                              # store final models
│   └── modelo_produccion.joblib        # el artefacto que sirve la demo
├── notebooks
│   ├── 1-data                          # data extraction and cleaning
│   ├── 2-exploration                   # exploratory data analysis (EDA)
│   ├── 3-analysis                      # Statistical analysis, hypothesis testing.
│   ├── 4-feat_eng                      # feature engineering (creation, selection, and transformation.)
│   ├── 5-models                        # model training, evaluation, and hyperparameter tuning.
│   ├── 6-interpretation                # model interpretation
│   ├── 7-deploy                        # model packaging, deployment strategies.
│   ├── 8-reports                       # story telling, summaries and analysis conclusions.
│   ├── notebook_template.ipynb         # template for notebooks
│   └── README.md                       # information about the notebooks
├── .pre-commit-config.yaml             # configuration for pre-commit hooks
├── pyproject.toml                      # dependencies for the python project
├── README.md                           # description of your project
├── src                                 # source code for use in this project
│   ├── README.md                       # description of src structure
│   ├── tmp_mock.py                     # example python file
│   ├── data                            # data extraction, validation, processing, transformation
│   │   ├── transformaciones.py         # transformaciones reutilizadas por los pipelines
│   │   ├── particion.py                # chequeos de la particion train/test (fuga y deriva)
│   │   ├── preprocesamiento.py         # pipeline de sklearn: imputacion, encoding y escalado
│   │   └── validacion.py               # motor de validacion: esquemas, reglas e informes
│   ├── model                           # model training, evaluation, validation, export
│   │   ├── heuristica.py               # modelo base: regla manual como estimador sklearn
│   │   └── validacion.py               # validacion cruzada, diagnostico y curva de aprendizaje
│   ├── inference                       # model prediction, serving, monitoring
│   └── pipelines                       # orchestration of pipelines
│       ├── feature_pipeline            # transforms raw data into features and labels
│       │   └── feature_pipeline.py     # raw csv -> data/04_feature/admisiones_features.parquet
│       ├── training_pipeline           # transforms features and labels into a model
│       │   └── train_pipeline.py       # features -> modelo entrenado + metricas
│       └── inference_pipeline          # takes features and a trained model for predictions
│           └── inference_pipeline.py  # modelo + aspirantes -> predicciones en 07_model_output
├── tests                               # test code for your project
│   ├── test_mock.py                    # example test file
│   ├── data                            # tests for data module
│   ├── model                           # tests for model module
│   ├── inference                       # tests for inference module
│   └── pipelines                       # tests for pipelines module
└── .vscode                             # vscode configuration
    ├── extensions.json                 # list of recommended extensions
    ├── launch.json                     # vscode launch configuration
    └── settings.json                   # vscode settings
```

## Credits

This project was generated from [@JoseRZapata]'s [data science project template] template.

## References

- [Config devcontainer with python and UV](https://tech.dentsusoken.com/entry/2023/05/02/Dev_Container%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E3%82%B9%E3%83%86%E3%83%83%E3%83%97%E3%83%90%E3%82%A4%E3%82%B9%E3%83%86%E3%83%83%E3%83%97%E3%81%A7%E4%BD%9C%E3%82%8BPython%E3%82%A2%E3%83%97%E3%83%AA%E3%82%B1)

---
[@JoseRZapata]: https://github.com/JoseRZapata

[bandit]: https://github.com/PyCQA/bandit
[codecov]: https://codecov.io/
[Cookiecutter]:https://cookiecutter.readthedocs.io/en/stable/
[coverage.py]: https://coverage.readthedocs.io/
[Cruft]: https://cruft.github.io/cruft/
[data science project template]: https://github.com/JoseRZapata/data-science-project-template
[Data structure]: https://github.com/JoseRZapata/data-science-project-template/blob/main/template-data-science-container/data/README.md
[Mypy]: http://mypy-lang.org/
[Notebook template]: template-data-science-container/notebooks/notebook_template.ipynb
[pre-commit]: https://pre-commit.com/
[Pull Request template]: template-data-science-container/.github/pull_request_template.md
[Pytest]: https://docs.pytest.org/en/latest/
[Ruff]: https://docs.astral.sh/ruff/
[UV]: https://docs.astral.sh/uv/
