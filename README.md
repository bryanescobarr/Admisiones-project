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

Formulario web donde un aspirante introduce su perfil académico y obtiene su probabilidad
estimada de admisión, junto con la clasificación de esa universidad como opción **segura**,
**probable** o **ambiciosa**, y el desglose de cuánto aportó cada dato de su perfil.

### Ejecutar en local

```bash
uv sync --all-extras --dev
uv run streamlit run app.py
```

Se abre en <http://localhost:8501>. No hace falta ejecutar ningún notebook antes: el modelo
entrenado ya está versionado en `data/06_models/modelo_final_automl.joblib`.

Para verificar que todo funciona sin abrir el navegador:

```bash
uv run pytest tests/test_app.py tests/test_prediccion.py -v
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
| `app.py` | interfaz: formulario, resultados, gráfico de aportes, advertencias |
| `src/inference/prediccion.py` | lógica: cargar el modelo, predecir, clasificar en cestas, explicar con SHAP |
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
│   │   └── validacion.py               # motor de validacion: esquemas, reglas e informes
│   ├── model                           # model training, evaluation, validation, export
│   ├── inference                       # model prediction, serving, monitoring
│   └── pipelines                       # orchestration of pipelines
│       ├── feature_pipeline            # transforms raw data into features and labels
│       │   └── feature_pipeline.py     # raw csv -> data/04_feature/admisiones_features.parquet
│       ├── training_pipeline           # transforms features and labels into a model
│       └── inference_pipeline          # takes features and a trained model for predictions
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
