"""Pipeline de preprocesamiento de scikit-learn, en código y no dentro de un notebook.

Reproduce la variante elegida con evidencia en `notebooks/4-feat_eng/04`: los siete
predictores originales, imputación por mediana en las numéricas y por moda en las
categóricas, `OrdinalEncoder` con las cinco categorías declaradas y `StandardScaler`.

**Por qué existe este módulo.** El pipeline ajustado está versionado en
`data/06_models/pipeline_preprocesamiento.joblib`, pero ese archivo lo produjo un
cuaderno: reentrenar dependiendo de él obliga a abrir Jupyter para regenerarlo y hace
imposible reproducir el modelo desde cero con un solo comando. Aquí está la *receta*; el
`.joblib` es una *instancia ya ajustada* de esa receta. `tests/data/test_preprocesamiento.py`
comprueba que ambos producen exactamente la misma salida.

**Por qué es un `Pipeline` y no una función que transforma el DataFrame.** Porque la
imputación y el escalado aprenden parámetros —la mediana, la media, la desviación— y
tienen que aprenderlos **solo del conjunto de entrenamiento**. Dentro de un `Pipeline`,
`fit` los aprende una vez y `transform` los aplica; calculados a mano sobre todo el
dataset, el modelo parecería mejor de lo que es.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
)

from data.transformaciones import a_tipos_sklearn, recortar_a_dominio

COLS_NUMERICAS: list[str] = ["gre_score", "toefl_score", "sop", "lor", "cgpa"]
COLS_ORDINALES: list[str] = ["university_rating"]
COLS_BOOLEANAS: list[str] = ["research"]

# el orden se declara a mano para fijar el mapeo: si en produccion solo llegan ratings 3 y
# 4, el encoder sigue codificandolos igual que en entrenamiento
CATEGORIAS_RATING: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0]


def construir_transformador_columnas(
    cols_numericas: list[str] | None = None,
    cols_ordinales: list[str] | None = None,
    cols_booleanas: list[str] | None = None,
) -> ColumnTransformer:
    """`ColumnTransformer` con un tratamiento por tipo de columna.

    - **numéricas**: mediana + estandarización.
    - **ordinal** (`university_rating`): moda + `OrdinalEncoder` con las categorías
      declaradas + estandarización.
    - **booleana** (`research`): moda, sin escalar (una binaria estandarizada pierde
      legibilidad sin ganar nada).
    """
    numericas = list(cols_numericas if cols_numericas is not None else COLS_NUMERICAS)
    ordinales = list(cols_ordinales if cols_ordinales is not None else COLS_ORDINALES)
    booleanas = list(cols_booleanas if cols_booleanas is not None else COLS_BOOLEANAS)

    tuberia_numerica = Pipeline(
        [("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]
    )
    tuberia_ordinal = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OrdinalEncoder(categories=[CATEGORIAS_RATING] * len(ordinales))),
            ("escalar", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", tuberia_numerica, numericas),
            ("ord", tuberia_ordinal, ordinales),
            ("bool", SimpleImputer(strategy="most_frequent"), booleanas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def construir_preprocesamiento(
    cols_numericas: list[str] | None = None,
    cols_ordinales: list[str] | None = None,
    cols_booleanas: list[str] | None = None,
) -> Pipeline:
    """Pipeline completo de datos crudos a matriz de atributos lista para el modelo.

    ```text
    tipos     FunctionTransformer  -> nullable de pandas a float64 con np.nan
    dominio   FunctionTransformer  -> recorta a los rangos documentados
    columnas  ColumnTransformer    -> imputa, codifica y escala segun el tipo
    ```

    Las dos primeras etapas usan funciones de `src/data/transformaciones.py`, importables
    desde un módulo: un `FunctionTransformer` guarda una *referencia* a la función, así que
    definida en un notebook el artefacto serializado sería inservible fuera de él.
    """
    pipeline = Pipeline(
        [
            ("tipos", FunctionTransformer(a_tipos_sklearn)),
            ("dominio", FunctionTransformer(recortar_a_dominio)),
            (
                "columnas",
                construir_transformador_columnas(cols_numericas, cols_ordinales, cols_booleanas),
            ),
        ]
    )
    # salida como DataFrame: conserva los nombres de las columnas hasta el modelo, que es lo
    # que permite explicar una prediccion con SHAP en `7-deploy` sin recordar el orden
    return pipeline.set_output(transform="pandas")
