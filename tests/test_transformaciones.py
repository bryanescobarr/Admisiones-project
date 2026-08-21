import numpy as np
import pandas as pd
import pytest

from data.transformaciones import (
    a_tipos_sklearn,
    agregar_atributos_derivados,
    recortar_a_dominio,
    transformaciones_no_lineales,
)


def _datos_ejemplo() -> pd.DataFrame:
    """Dos filas con los tipos nullable que produce el parquet de 02_intermediate."""
    return pd.DataFrame(
        {
            "gre_score": pd.array([340, None], dtype="Int64"),
            "toefl_score": pd.array([120, 100], dtype="Int64"),
            "university_rating": pd.Categorical(
                ["5", "3"], categories=["1", "2", "3", "4", "5"], ordered=True
            ),
            "sop": pd.array([4.0, 2.5], dtype="Float64"),
            "lor": pd.array([5.0, 3.0], dtype="Float64"),
            "cgpa": pd.array([9.5, 8.0], dtype="Float64"),
            "research": pd.array([True, None], dtype="boolean"),
        }
    )


def test_a_tipos_sklearn_convierte_todo_a_float_con_nan() -> None:
    """Los dtypes nullable pasan a float64, pd.NA pasa a np.nan y el booleano a 0/1."""
    convertido = a_tipos_sklearn(_datos_ejemplo())

    assert (convertido.dtypes == "float64").all()
    assert convertido.loc[0].tolist() == [340.0, 120.0, 5.0, 4.0, 5.0, 9.5, 1.0]
    assert convertido.loc[1].isna().tolist() == [True, False, False, False, False, False, True]


def test_recortar_a_dominio_limita_los_valores_imposibles() -> None:
    """Un valor fuera del dominio documentado se recorta; uno válido no se toca."""
    datos = pd.DataFrame({"gre_score": [900.0, 310.0], "cgpa": [-1.0, 8.0]})

    recortado = recortar_a_dominio(datos)

    assert recortado["gre_score"].tolist() == [340.0, 310.0]
    assert recortado["cgpa"].tolist() == [0.0, 8.0]


def test_recortar_a_dominio_ignora_columnas_desconocidas() -> None:
    """Una columna que no está en DOMINIOS pasa sin cambios."""
    datos = pd.DataFrame({"otra_columna": [1000.0]})

    assert recortar_a_dominio(datos)["otra_columna"].tolist() == [1000.0]


def test_agregar_atributos_derivados() -> None:
    """Los tres atributos derivados se calculan con las fórmulas documentadas."""
    derivado = agregar_atributos_derivados(a_tipos_sklearn(_datos_ejemplo()))

    esperado = {
        # (cgpa 9.5/10 + gre normalizado 1.0 + toefl 120/120) / 3
        "indice_academico": pytest.approx((0.95 + 1.0 + 1.0) / 3),
        "sop_lor_media": pytest.approx((4.0 + 5.0) / 2),
        "rating_x_research": pytest.approx(5.0 * 1.0),
    }
    assert derivado.loc[0, list(esperado)].to_dict() == esperado


def test_transformaciones_no_lineales() -> None:
    """Se añaden las tres columnas sin alterar las originales."""
    datos = a_tipos_sklearn(_datos_ejemplo())

    ampliado = transformaciones_no_lineales(datos)

    nuevas = ["log_cgpa", "sqrt_sop", "gre_score_cuadrado"]
    assert ampliado.loc[0, nuevas].tolist() == [np.log1p(9.5), np.sqrt(4.0), 340.0**2]
    assert ampliado[datos.columns].equals(datos)
