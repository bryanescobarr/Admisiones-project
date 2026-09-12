import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from inference import prediccion
from inference.prediccion import (
    COLUMNAS_ENTRADA,
    CORTES_CESTA,
    MAE_DE_RESPALDO,
    MAE_MODELO,
    MODELO_EN_METRICAS,
    MODELO_SERVIDO,
    PREDICCION_MINIMA,
    PREDICCION_MINIMA_DE_RESPALDO,
    cargar_modelo,
    clasificar_cesta,
    construir_fila,
    contribuciones,
    intervalo_estimado,
    predecir,
    prediccion_media,
    ruta_modelo_por_defecto,
)
from pipelines.feature_pipeline.feature_pipeline import (
    ejecutar_pipeline as ejecutar_feature_pipeline,
)
from pipelines.training_pipeline.train_pipeline import separar_train_test

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


# --- Modelo de servicio y constantes -----------------------------------------------------


def _informe_de_metricas(mae: float = 0.0500, minima: float | None = None) -> pd.DataFrame:
    """Informe con la forma del que escribe el training pipeline."""
    fila = {"modelo": MODELO_EN_METRICAS, "MAE": mae, "R2": 0.79}
    if minima is not None:
        fila["prediccion_minima"] = minima
    return pd.DataFrame([fila, {"modelo": "dummy", "MAE": 0.1191, "R2": -0.0017}])


def test_la_demo_sirve_el_artefacto_del_training_pipeline() -> None:
    """La ruta por defecto apunta al modelo de servicio, no al `.joblib` del notebook."""
    ruta = ruta_modelo_por_defecto()

    assert ruta.name == MODELO_SERVIDO == "modelo_produccion.joblib"
    assert ruta.parent.name == "models"
    assert "data" not in ruta.parts, "el artefacto de servicio ya no vive bajo data/"
    assert ruta.exists(), "el artefacto de servicio tiene que estar versionado"


def test_el_modelo_servido_se_carga_y_predice() -> None:
    """Cargarlo no basta: tiene que producir una probabilidad utilizable."""
    prediccion = predecir(cargar_modelo(), PERFIL_FUERTE)

    assert 0.0 <= prediccion <= 1.0


def test_si_falta_el_modelo_el_error_cita_el_script_que_lo_genera(tmp_path: Path) -> None:
    """Quien lea el error tiene que saber qué ejecutar: un script, no un notebook."""
    with pytest.raises(FileNotFoundError, match=re.escape("train_pipeline.py")) as error:
        cargar_modelo(tmp_path / "no_existe.joblib")

    assert "notebook" not in str(error.value).lower()


def test_las_constantes_se_leen_del_informe_del_training_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La cifra que ve el usuario sale de la ejecución que generó el artefacto."""
    ruta = tmp_path / "metricas_training_pipeline.parquet"
    _informe_de_metricas(mae=0.0500, minima=0.4100).to_parquet(ruta, index=False)
    monkeypatch.setattr(prediccion, "ruta_metricas_por_defecto", lambda: ruta)

    assert prediccion.metrica_del_modelo_servido("MAE", MAE_DE_RESPALDO) == pytest.approx(0.05)
    assert prediccion.metrica_del_modelo_servido(
        "prediccion_minima", PREDICCION_MINIMA_DE_RESPALDO
    ) == pytest.approx(0.41)


def test_la_metrica_leida_es_un_float(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El tipo importa: la interfaz las formatea como números, no como celdas de pandas."""
    ruta = tmp_path / "metricas.parquet"
    _informe_de_metricas().to_parquet(ruta, index=False)
    monkeypatch.setattr(prediccion, "ruta_metricas_por_defecto", lambda: ruta)

    assert isinstance(prediccion.metrica_del_modelo_servido("MAE", MAE_DE_RESPALDO), float)
    assert isinstance(MAE_MODELO, float)
    assert isinstance(PREDICCION_MINIMA, float)


def test_sin_informe_de_metricas_se_usa_el_valor_medido(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """En el despliegue no hay reportes —`data/**` no se versiona— y la demo debe arrancar."""
    monkeypatch.setattr(prediccion, "ruta_metricas_por_defecto", lambda: tmp_path / "no.parquet")

    assert prediccion.metrica_del_modelo_servido("MAE", MAE_DE_RESPALDO) == MAE_DE_RESPALDO


def test_si_el_informe_no_trae_la_columna_se_usa_el_valor_medido(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hoy el informe no incluye la predicción mínima: el respaldo cubre ese hueco."""
    ruta = tmp_path / "metricas.parquet"
    _informe_de_metricas().to_parquet(ruta, index=False)
    monkeypatch.setattr(prediccion, "ruta_metricas_por_defecto", lambda: ruta)

    assert (
        prediccion.metrica_del_modelo_servido("prediccion_minima", PREDICCION_MINIMA_DE_RESPALDO)
        == PREDICCION_MINIMA_DE_RESPALDO
    )


def test_si_el_modelo_no_esta_en_el_informe_se_usa_el_valor_medido(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un informe de otra ejecución no debe colar la métrica de un modelo distinto."""
    ruta = tmp_path / "metricas.parquet"
    pd.DataFrame([{"modelo": "otro_modelo", "MAE": 0.2}]).to_parquet(ruta, index=False)
    monkeypatch.setattr(prediccion, "ruta_metricas_por_defecto", lambda: ruta)

    assert prediccion.metrica_del_modelo_servido("MAE", MAE_DE_RESPALDO) == MAE_DE_RESPALDO


def test_las_cifras_que_muestra_la_demo_son_las_del_modelo_servido(tmp_path: Path) -> None:
    """Se recalculan desde los datos crudos versionados, sobre la misma partición.

    Es la prueba que impide que las constantes se queden describiendo a otro modelo: si
    alguien regenera el artefacto y las cifras cambian, esto lo dice.
    """
    raiz = Path(__file__).resolve().parents[1]
    features = ejecutar_feature_pipeline(
        raiz / "data" / "01_raw" / "Admission_Predict.csv", tmp_path / "features.parquet"
    )
    particion = separar_train_test(features)

    prediccion_test = np.clip(cargar_modelo().predict(particion.X_test), 0.0, 1.0)

    assert mean_absolute_error(particion.y_test, prediccion_test) == pytest.approx(
        MAE_MODELO, abs=1e-4
    )
    assert float(prediccion_test.min()) == pytest.approx(PREDICCION_MINIMA, abs=1e-3)
