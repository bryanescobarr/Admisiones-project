import numpy as np
import pandas as pd
import pytest

from model.heuristica import HeuristicaAdmision


def _aspirantes() -> pd.DataFrame:
    """Tres perfiles: máximo posible, mínimo posible e intermedio con un dato faltante."""
    return pd.DataFrame(
        {
            "gre_score": pd.array([340, 260, None], dtype="Int64"),
            "toefl_score": pd.array([120, 0, 110], dtype="Int64"),
            "university_rating": [5.0, 1.0, 4.0],
            "sop": [5.0, 1.0, 4.0],
            "lor": [5.0, 1.0, 4.0],
            "cgpa": [10.0, 0.0, 9.0],
            "research": [1.0, 0.0, 1.0],
        }
    )


def test_sin_recalibrar_devuelve_el_puntaje_normalizado() -> None:
    """El perfil máximo puntúa 1.0 y el mínimo 0.0."""
    modelo = HeuristicaAdmision(recalibrar=False).fit(_aspirantes())

    prediccion = modelo.predict(_aspirantes())

    assert prediccion[0] == pytest.approx(1.0)
    assert prediccion[1] == pytest.approx(0.0)


def test_ignora_los_valores_faltantes() -> None:
    """Una fila con un dato ausente promedia las columnas que sí tiene, sin NaN."""
    modelo = HeuristicaAdmision(recalibrar=False).fit(_aspirantes())

    prediccion = modelo.predict(_aspirantes())

    assert not np.isnan(prediccion).any()
    # promedio de las 6 columnas presentes: (110/120 + 3/4 + 3/4 + 3/4 + 0.9 + 1) / 6
    assert prediccion[2] == pytest.approx((110 / 120 + 0.75 * 3 + 0.9 + 1.0) / 6)


def test_la_recalibracion_ajusta_la_escala_al_objetivo() -> None:
    """Con recalibración, la predicción se acerca al objetivo observado."""
    datos = _aspirantes()
    objetivo = np.array([0.95, 0.35, 0.80])

    modelo = HeuristicaAdmision(recalibrar=True).fit(datos, objetivo)
    prediccion = modelo.predict(datos)

    assert np.abs(prediccion - objetivo).mean() < np.abs(objetivo.mean() - objetivo).mean()
    assert ((prediccion >= 0) & (prediccion <= 1)).all()


def test_falla_si_no_hay_columnas_conocidas() -> None:
    """Un DataFrame sin ninguna de las columnas esperadas no se puede puntuar."""
    with pytest.raises(ValueError, match="Ninguna columna conocida"):
        HeuristicaAdmision(recalibrar=False).fit(pd.DataFrame({"otra": [1.0]}))
