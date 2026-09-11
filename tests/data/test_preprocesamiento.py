from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from data.preprocesamiento import (
    COLS_BOOLEANAS,
    COLS_NUMERICAS,
    COLS_ORDINALES,
    construir_preprocesamiento,
)

COLUMNAS_SALIDA = [*COLS_NUMERICAS, *COLS_ORDINALES, *COLS_BOOLEANAS]
TOLERANCIA = 1e-6


def _datos_crudos(filas: int = 40) -> pd.DataFrame:
    """Perfiles sintéticos con los tipos nullable que produce el feature pipeline."""
    generador = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "gre_score": generador.integers(290, 341, filas).astype("float64"),
            "toefl_score": generador.integers(95, 121, filas).astype("float64"),
            "university_rating": generador.integers(1, 6, filas).astype("float64"),
            "sop": generador.integers(2, 11, filas) / 2,
            "lor": generador.integers(2, 11, filas) / 2,
            "cgpa": generador.uniform(6.8, 9.9, filas).round(2),
            "research": generador.integers(0, 2, filas).astype("float64"),
        }
    )


def test_la_salida_conserva_los_nombres_de_las_columnas() -> None:
    """El pipeline devuelve un DataFrame: los nombres llegan hasta el modelo."""
    datos = _datos_crudos()

    preparado = construir_preprocesamiento().fit_transform(datos)

    assert isinstance(preparado, pd.DataFrame)
    assert list(preparado.columns) == COLUMNAS_SALIDA


def test_las_numericas_quedan_estandarizadas_y_la_binaria_no() -> None:
    """Media 0 y desviación 1 en las numéricas; `research` conserva sus 0 y 1."""
    preparado = construir_preprocesamiento().fit_transform(_datos_crudos(100))

    assert preparado[COLS_NUMERICAS].mean().abs().max() < TOLERANCIA
    assert abs(preparado[COLS_NUMERICAS].std(ddof=0).max() - 1) < TOLERANCIA
    assert set(preparado["research"].unique()) <= {0.0, 1.0}


def test_los_nulos_se_imputan_con_la_mediana_aprendida_en_entrenamiento() -> None:
    """La estadística sale de *train*: es la única forma de no filtrar información."""
    entrenamiento = _datos_crudos(60)
    preprocesador = construir_preprocesamiento().fit(entrenamiento)
    nuevo = entrenamiento.head(1).copy()
    nuevo.loc[:, "cgpa"] = np.nan

    preparado = preprocesador.transform(nuevo)

    mediana_train = float(entrenamiento["cgpa"].median())
    esperado = preprocesador.transform(entrenamiento.head(1).assign(cgpa=mediana_train))
    assert preparado["cgpa"].iloc[0] == pytest.approx(esperado["cgpa"].iloc[0])
    assert not preparado.isna().any().any()


def test_recorta_los_valores_imposibles_antes_de_escalar() -> None:
    """Un GRE de 900 en inferencia se recorta al dominio en vez de llegar al modelo."""
    preprocesador = construir_preprocesamiento().fit(_datos_crudos(60))
    extremo = _datos_crudos(1).assign(gre_score=900.0, cgpa=15.0)

    preparado = preprocesador.transform(extremo)
    tope = preprocesador.transform(_datos_crudos(1).assign(gre_score=340.0, cgpa=10.0))

    assert preparado["gre_score"].iloc[0] == pytest.approx(tope["gre_score"].iloc[0])
    assert preparado["cgpa"].iloc[0] == pytest.approx(tope["cgpa"].iloc[0])


def test_reproduce_el_pipeline_ajustado_en_el_notebook() -> None:
    """La receta en código y el artefacto de `4-feat_eng` producen la misma salida.

    Es la prueba que justifica haber reescrito el preprocesamiento fuera del cuaderno: si
    alguien cambia un paso aquí y no allí, la divergencia se ve en esta prueba y no en las
    predicciones de la demo.
    """
    raiz = Path(__file__).resolve().parents[2]
    artefacto = raiz / "data" / "06_models" / "pipeline_preprocesamiento.joblib"
    datos = raiz / "data" / "02_intermediate" / "admisiones_type_fixed.parquet"
    if not artefacto.exists() or not datos.exists():
        pytest.skip("Faltan los artefactos generados por los notebooks 02 y 04")

    df = pd.read_parquet(datos)
    X = df.drop(columns=["chance_of_admit"])
    y = df["chance_of_admit"].astype(float)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)

    en_codigo = construir_preprocesamiento().fit(X_train, y_train).transform(X_test)
    del_notebook = clone(joblib.load(artefacto)).fit(X_train, y_train).transform(X_test)

    assert list(en_codigo.columns) == list(del_notebook.columns)
    assert np.allclose(en_codigo.to_numpy(), del_notebook.to_numpy())
