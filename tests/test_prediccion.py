import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from inference.prediccion import (
    COLUMNAS_ENTRADA,
    CORTES_CESTA,
    cargar_modelo,
    clasificar_cesta,
    construir_fila,
    contribuciones,
    intervalo_estimado,
    predecir,
    prediccion_media,
)

PERFIL_FUERTE = {
    "gre_score": 335,
    "toefl_score": 118,
    "university_rating": 5.0,
    "sop": 5.0,
    "lor": 4.5,
    "cgpa": 9.6,
    "research": 1.0,
}
PERFIL_DEBIL = {
    "gre_score": 295,
    "toefl_score": 95,
    "university_rating": 1.0,
    "sop": 2.0,
    "lor": 2.0,
    "cgpa": 7.2,
    "research": 0.0,
}


@pytest.fixture(scope="module")
def modelo() -> Pipeline:
    """El artefacto entrenado, cargado una sola vez para todas las pruebas."""
    return cargar_modelo()


def test_construir_fila_respeta_el_orden_y_marca_los_ausentes() -> None:
    """Las columnas van en el orden del entrenamiento y lo que falta queda como NaN."""
    promedio = 9.0
    fila = construir_fila({"cgpa": promedio})

    assert list(fila.columns) == COLUMNAS_ENTRADA
    assert fila.loc[0, "cgpa"] == promedio
    assert fila.drop(columns=["cgpa"]).isna().all(axis=None)


def test_clasificar_cesta_en_los_bordes() -> None:
    """Los cortes son inclusivos por abajo, como en el análisis de 03.3."""
    corte_bajo, corte_alto = CORTES_CESTA

    assert clasificar_cesta(corte_bajo - 0.01) == "ambiciosa"
    assert clasificar_cesta(corte_bajo) == "probable"
    assert clasificar_cesta(corte_alto - 0.01) == "probable"
    assert clasificar_cesta(corte_alto) == "segura"


def test_un_perfil_fuerte_predice_mas_que_uno_debil(modelo: Pipeline) -> None:
    """La prueba de humo que importa: el modelo ordena los perfiles como se espera."""
    fuerte = predecir(modelo, PERFIL_FUERTE)
    debil = predecir(modelo, PERFIL_DEBIL)

    assert fuerte > debil
    assert 0.0 <= debil <= fuerte <= 1.0
    assert clasificar_cesta(fuerte) == "segura"


def test_predice_aunque_falten_datos(modelo: Pipeline) -> None:
    """Un aspirante puede consultar sin tener todos los datos: el pipeline imputa."""
    incompleto = {clave: valor for clave, valor in PERFIL_FUERTE.items() if clave != "toefl_score"}

    prediccion = predecir(modelo, incompleto)

    assert not np.isnan(prediccion)
    assert 0.0 <= prediccion <= 1.0


def test_el_intervalo_se_mantiene_dentro_del_rango_valido() -> None:
    """Una probabilidad no puede salirse de [0, 1] ni con el margen de error sumado."""
    assert intervalo_estimado(0.98)[1] <= 1.0
    assert intervalo_estimado(0.01)[0] >= 0.0


def test_las_contribuciones_explican_la_prediccion(modelo: Pipeline) -> None:
    """SHAP es aditivo: base + aportes debe reconstruir exactamente la predicción."""
    aportes = contribuciones(modelo, PERFIL_FUERTE)
    reconstruida = prediccion_media(modelo) + aportes.sum()

    assert set(aportes.index) == set(COLUMNAS_ENTRADA)
    assert reconstruida == pytest.approx(predecir(modelo, PERFIL_FUERTE), abs=1e-6)
    assert isinstance(aportes, pd.Series)
