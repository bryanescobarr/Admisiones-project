"""Transformaciones de datos reutilizables por los pipelines de scikit-learn.

Estas funciones viven en `src/` y no dentro del notebook a propósito: un
`FunctionTransformer` guarda una *referencia* a la función, no su código, así que un
pipeline serializado con `joblib` solo se puede volver a cargar si la función es
importable desde un módulo. Definidas en el notebook, el artefacto quedaría inservible
fuera de él.
"""

import numpy as np
import pandas as pd

# dominio documentado de cada columna, tomado de data/01_raw/Informacion.txt
DOMINIOS: dict[str, tuple[float, float]] = {
    "gre_score": (0, 340),
    "toefl_score": (0, 120),
    "university_rating": (1, 5),
    "sop": (1, 5),
    "lor": (1, 5),
    "cgpa": (0, 10),
    "research": (0, 1),
}


def a_tipos_sklearn(datos: pd.DataFrame) -> pd.DataFrame:
    """Convierte los tipos nullable de pandas a `float64` con `np.nan`.

    scikit-learn no entiende `pd.NA` ni los dtypes `Int64`/`boolean`/`category`:
    necesita valores compatibles con numpy. Esta es la primera etapa de cualquier
    pipeline que parta del parquet de `02_intermediate`.

    >>> a_tipos_sklearn(pd.DataFrame({"research": pd.array([True, None], dtype="boolean")}))
       research
    0       1.0
    1       NaN
    """
    convertidos = datos.copy()
    for columna in convertidos.columns:
        convertidos[columna] = pd.to_numeric(convertidos[columna], errors="coerce").astype(
            "float64"
        )
    return convertidos


def recortar_a_dominio(datos: pd.DataFrame) -> pd.DataFrame:
    """Recorta cada columna a su dominio documentado.

    No cambia ningún valor del dataset actual (ya se verificó que todos están dentro de
    rango); actúa como red de seguridad en inferencia, cuando llegue un dato nuevo
    imposible —un GRE de 900 por un error de captura— que de otro modo se propagaría al
    modelo.
    """
    recortados = datos.copy()
    for columna, (minimo, maximo) in DOMINIOS.items():
        if columna in recortados.columns:
            recortados[columna] = recortados[columna].clip(lower=minimo, upper=maximo)
    return recortados


def agregar_atributos_derivados(datos: pd.DataFrame) -> pd.DataFrame:
    """Añade los atributos derivados propuestos por el análisis de `3-analysis`.

    - `indice_academico`: promedio de `cgpa`, `gre_score` y `toefl_score` normalizados.
      Las tres correlacionan entre 0.83 y 0.85 (VIF hasta 5.06): resumen un mismo rasgo
      latente de desempeño académico.
    - `sop_lor_media`: promedio de las dos evaluaciones cualitativas del expediente
      (r = 0.74 entre ellas).
    - `rating_x_research`: interacción entre `university_rating` y `research`, cuyo
      efecto conjunto crece con el nivel de la universidad (+0.04 en el nivel 1,
      +0.18 en el 4 y el 5).
    """
    derivados = datos.copy()
    gre_normalizado = (derivados["gre_score"] - 260) / (340 - 260)
    derivados["indice_academico"] = (
        derivados["cgpa"] / 10 + gre_normalizado + derivados["toefl_score"] / 120
    ) / 3
    derivados["sop_lor_media"] = (derivados["sop"] + derivados["lor"]) / 2
    derivados["rating_x_research"] = derivados["university_rating"] * derivados["research"]
    return derivados


def transformaciones_no_lineales(datos: pd.DataFrame) -> pd.DataFrame:
    """Añade `log(cgpa)`, `sqrt(sop)` y `gre_score^2` como atributos adicionales.

    Se incluye para poder **medir** si aportan algo. El análisis univariable mostró
    distribuciones simétricas (|skew| < 0.35), así que la expectativa es que no mejoren:
    las transformaciones de forma sirven para corregir asimetría, y aquí no la hay.
    """
    ampliados = datos.copy()
    ampliados["log_cgpa"] = np.log1p(ampliados["cgpa"])
    ampliados["sqrt_sop"] = np.sqrt(ampliados["sop"])
    ampliados["gre_score_cuadrado"] = ampliados["gre_score"] ** 2
    return ampliados
