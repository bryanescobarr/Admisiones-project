from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from data.particion import Particion
from data.validacion import ErrorValidacion
from pipelines.feature_pipeline.feature_pipeline import TARGET
from pipelines.training_pipeline.train_pipeline import (
    CATALOGO_MODELOS,
    RutasEntrenamiento,
    chequear_particion,
    construir_informe,
    construir_modelo,
    ejecutar_pipeline,
    entrenar,
    evaluar,
    guardar_metricas,
    guardar_modelo,
    leer_features,
    main,
    separar_train_test,
    validar_en_entrenamiento,
)

FILAS = 80
PROPORCION_TEST = 0.25
SEMILLA = 0
TOLERANCIA = 1e-9
CORRELACION_MINIMA = 0.5


def _features(filas: int = FILAS) -> pd.DataFrame:
    """Features sintéticas que cumplen el contrato del feature pipeline.

    El objetivo se construye a partir de los predictores más un poco de ruido, para que un
    modelo pueda aprender algo y las métricas tengan sentido.
    """
    generador = np.random.default_rng(SEMILLA)
    datos = pd.DataFrame(
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
    senal = (
        0.6 * (datos["cgpa"] - 6.8) / 3.1
        + 0.2 * (datos["gre_score"] - 290) / 50
        + 0.1 * datos["research"]
    )
    datos[TARGET] = (0.3 + senal + generador.normal(0, 0.01, filas)).clip(0.05, 0.99).round(4)
    return datos


@pytest.fixture
def parquet_features(tmp_path: Path) -> Path:
    """Archivo de features en disco, como el que deja el feature pipeline."""
    ruta = tmp_path / "admisiones_features.parquet"
    _features().to_parquet(ruta, index=False)
    return ruta


# --- Lectura de los datos ---------------------------------------------------------------


def test_leer_features_carga_el_parquet_del_paso_anterior(parquet_features: Path) -> None:
    """La entrada del training pipeline es la salida del feature pipeline."""
    features = leer_features(parquet_features)

    assert len(features) == FILAS
    assert TARGET in features.columns


def test_leer_features_falla_con_un_mensaje_util_si_no_existe(tmp_path: Path) -> None:
    """El mensaje dice qué ejecutar antes, que es lo que necesita quien lo lee."""
    with pytest.raises(FileNotFoundError, match="feature_pipeline"):
        leer_features(tmp_path / "no_existe.parquet")


def test_leer_features_rechaza_un_archivo_que_no_cumple_el_contrato(tmp_path: Path) -> None:
    """Nada garantiza que el parquet lo generara la version actual del feature pipeline."""
    ruta = tmp_path / "corruptas.parquet"
    features = _features()
    features.loc[0, "cgpa"] = 42.0
    features.to_parquet(ruta, index=False)

    with pytest.raises(ErrorValidacion, match="fuera_de_rango"):
        leer_features(ruta)


# --- Separacion train / test ------------------------------------------------------------


def test_separar_train_test_reparte_las_filas_sin_solaparlas() -> None:
    """Una fila de prueba que estuvo en entrenamiento invalida toda la evaluación."""
    X_train, X_test, y_train, y_test = separar_train_test(_features(), PROPORCION_TEST, SEMILLA)

    assert len(X_test) == int(FILAS * PROPORCION_TEST)
    assert len(X_train) + len(X_test) == FILAS
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(y_train) == len(X_train) and len(y_test) == len(X_test)


def test_separar_train_test_deja_el_objetivo_fuera_de_los_predictores() -> None:
    """Con el objetivo dentro de X, cualquier modelo sacaría un error de cero."""
    X_train, _, y_train, _ = separar_train_test(_features(), PROPORCION_TEST, SEMILLA)

    assert TARGET not in X_train.columns
    assert y_train.dtype == "float64"


def test_la_particion_es_reproducible_con_la_misma_semilla() -> None:
    """Sin reproducibilidad, dos ejecuciones no son comparables."""
    primera = separar_train_test(_features(), PROPORCION_TEST, SEMILLA).X_train
    segunda = separar_train_test(_features(), PROPORCION_TEST, SEMILLA).X_train

    assert primera.index.tolist() == segunda.index.tolist()


# --- Construccion y entrenamiento del modelo --------------------------------------------


def test_el_catalogo_arma_un_pipeline_con_preprocesamiento() -> None:
    """Los modelos que consumen la matriz preparada llevan el paso de preparación."""
    modelo = construir_modelo("ridge", SEMILLA)

    assert list(modelo.named_steps) == ["preparacion", "modelo"]


def test_la_heuristica_consume_los_datos_crudos() -> None:
    """Normaliza por el dominio documentado: escalarla antes no tendría sentido."""
    modelo = construir_modelo("heuristica", SEMILLA)

    assert list(modelo.named_steps) == ["modelo"]


def test_un_modelo_desconocido_falla_con_las_opciones_disponibles() -> None:
    """El mensaje enumera el catálogo en vez de dejar un KeyError."""
    with pytest.raises(ValueError, match="Disponibles"):
        construir_modelo("red_neuronal_magica", SEMILLA)


def test_entrenar_ajusta_el_pipeline_completo() -> None:
    """Tras entrenar, el pipeline predice sobre datos nuevos sin volver a ajustarse."""
    X_train, X_test, y_train, _ = separar_train_test(_features(), PROPORCION_TEST, SEMILLA)

    modelo = entrenar(construir_modelo("ridge", SEMILLA), X_train, y_train)
    prediccion = modelo.predict(X_test)

    assert len(prediccion) == len(X_test)
    assert np.isfinite(prediccion).all()


# --- Metricas ---------------------------------------------------------------------------


def test_evaluar_devuelve_las_cinco_metricas() -> None:
    """MAE y RMSE para el error, R2 y MAPE para comunicar, spearman para el ranking."""
    X_train, X_test, y_train, y_test = separar_train_test(_features(), PROPORCION_TEST, SEMILLA)
    modelo = entrenar(construir_modelo("ridge", SEMILLA), X_train, y_train)

    metricas = evaluar(modelo, X_test, y_test, "ridge")

    assert set(metricas) == {"modelo", "MAE", "RMSE", "R2", "MAPE", "spearman"}
    assert 0 < metricas["MAE"] < 1
    assert metricas["RMSE"] >= metricas["MAE"]
    assert metricas["spearman"] > CORRELACION_MINIMA


def test_un_modelo_que_predice_siempre_lo_mismo_no_tiene_correlacion_de_rangos() -> None:
    """El dummy no ordena nada: spearman no esta definida y se reporta como nula."""
    X_train, X_test, y_train, y_test = separar_train_test(_features(), PROPORCION_TEST, SEMILLA)
    modelo = entrenar(construir_modelo("dummy", SEMILLA), X_train, y_train)

    metricas = evaluar(modelo, X_test, y_test, "dummy")

    assert np.isnan(metricas["spearman"])


def test_el_modelo_entrenado_supera_a_la_referencia_trivial() -> None:
    """Si no le gana a predecir la media, no hay nada que desplegar."""
    X_train, X_test, y_train, y_test = separar_train_test(_features(), PROPORCION_TEST, SEMILLA)
    ridge = entrenar(construir_modelo("ridge", SEMILLA), X_train, y_train)
    dummy = entrenar(construir_modelo("dummy", SEMILLA), X_train, y_train)

    assert (
        evaluar(ridge, X_test, y_test, "ridge")["MAE"]
        < evaluar(dummy, X_test, y_test, "dummy")["MAE"]
    )


def test_validar_en_entrenamiento_mide_la_brecha_de_sobreajuste() -> None:
    """La diferencia entre entrenamiento y validación es el diagnóstico, no el adorno."""
    X_train, _, y_train, _ = separar_train_test(_features(), PROPORCION_TEST, SEMILLA)
    modelo = construir_modelo("ridge", SEMILLA)

    diagnostico = validar_en_entrenamiento(modelo, X_train, y_train, SEMILLA, pliegues=3)

    assert set(diagnostico) == {"MAE_cv", "MAE_cv_desv", "MAE_entrenamiento", "brecha_train_cv"}
    assert diagnostico["MAE_cv"] > 0
    assert diagnostico["brecha_train_cv"] == pytest.approx(
        diagnostico["MAE_cv"] - diagnostico["MAE_entrenamiento"], abs=TOLERANCIA
    )


def test_el_informe_ordena_por_error_y_calcula_la_mejora() -> None:
    """La tabla se lee de arriba abajo: el mejor modelo primero."""
    informe = construir_informe(
        [
            {"modelo": "ridge", "MAE": 0.05},
            {"modelo": "dummy", "MAE": 0.10},
        ]
    )

    assert informe["modelo"].tolist() == ["ridge", "dummy"]
    assert informe.loc[0, "mejora_vs_dummy_%"] == pytest.approx(50.0)


# --- Almacenamiento ---------------------------------------------------------------------


def test_guardar_modelo_verifica_que_el_artefacto_se_puede_recargar(tmp_path: Path) -> None:
    """Un artefacto que no se vuelve a cargar no es un modelo, es un archivo."""
    X_train, X_test, y_train, _ = separar_train_test(_features(), PROPORCION_TEST, SEMILLA)
    modelo = entrenar(construir_modelo("ridge", SEMILLA), X_train, y_train)
    ruta = tmp_path / "modelos" / "modelo.joblib"

    guardar_modelo(modelo, ruta, X_test)

    assert ruta.exists()
    assert np.allclose(joblib.load(ruta).predict(X_test), modelo.predict(X_test))


def test_guardar_metricas_escribe_un_parquet_relegible(tmp_path: Path) -> None:
    """Las métricas viven junto al resto de reportes del proyecto."""
    metricas = construir_informe([{"modelo": "ridge", "MAE": 0.05}])
    ruta = tmp_path / "reportes" / "metricas.parquet"

    guardar_metricas(metricas, ruta)

    assert pd.read_parquet(ruta).equals(metricas)


# --- Ejecucion completa -----------------------------------------------------------------


def test_ejecutar_pipeline_entrena_evalua_y_guarda(parquet_features: Path, tmp_path: Path) -> None:
    """El recorrido completo: features en disco -> modelo y métricas en disco."""
    ruta_modelo = tmp_path / "modelo.joblib"
    ruta_metricas = tmp_path / "metricas.parquet"

    resultado = ejecutar_pipeline(
        RutasEntrenamiento(parquet_features, ruta_modelo, ruta_metricas),
        nombre_modelo="ridge",
        semilla=SEMILLA,
    )

    assert ruta_modelo.exists()
    assert ruta_metricas.exists()
    assert resultado.metricas["modelo"].tolist() == ["ridge", "heuristica", "dummy"]
    assert resultado.metricas["MAE"].is_monotonic_increasing


def test_ejecutar_pipeline_puede_medir_solo_el_modelo_elegido(
    parquet_features: Path, tmp_path: Path
) -> None:
    """Sin referencias, la tabla trae una sola fila."""
    resultado = ejecutar_pipeline(
        RutasEntrenamiento(
            parquet_features, tmp_path / "modelo.joblib", tmp_path / "metricas.parquet"
        ),
        nombre_modelo="ridge",
        semilla=SEMILLA,
        con_referencias=False,
    )

    assert len(resultado.metricas) == 1


def test_main_ejecuta_el_script_desde_la_linea_de_comandos(
    parquet_features: Path, tmp_path: Path
) -> None:
    """La forma en que se ejecuta de forma autonoma."""
    ruta_modelo = tmp_path / "modelo.joblib"
    ruta_metricas = tmp_path / "metricas.parquet"

    codigo = main(
        [
            "--features",
            str(parquet_features),
            "--modelo",
            "ridge",
            "--modelo-salida",
            str(ruta_modelo),
            "--metricas",
            str(ruta_metricas),
        ]
    )

    assert codigo == 0
    assert ruta_modelo.exists()
    assert not pd.read_parquet(ruta_metricas).empty


def test_main_devuelve_codigo_de_error_si_faltan_las_features(tmp_path: Path) -> None:
    """Un orquestador necesita un codigo de salida, no un traceback."""
    ruta_modelo = tmp_path / "modelo.joblib"

    codigo = main(
        [
            "--features",
            str(tmp_path / "no_existe.parquet"),
            "--modelo-salida",
            str(ruta_modelo),
            "--metricas",
            str(tmp_path / "metricas.parquet"),
        ]
    )

    assert codigo == 1
    assert not ruta_modelo.exists()


def test_el_catalogo_documenta_cada_modelo() -> None:
    """Cada opción del catálogo dice qué es y de qué cuaderno sale."""
    for nombre, definicion in CATALOGO_MODELOS.items():
        assert definicion["descripcion"], f"{nombre} sin descripcion"
        assert callable(definicion["crear"])


# --- Chequeos de la particion train/test ------------------------------------------------


def test_chequear_particion_devuelve_el_informe_completo() -> None:
    """El pipeline deja constancia de cada chequeo, no solo de los que fallan."""
    particion = separar_train_test(_features(400), PROPORCION_TEST, SEMILLA)

    informe = chequear_particion(particion)

    assert set(informe["severidad"]) == {"ok"}
    assert "indices_solapados" in informe["chequeo"].tolist()


def test_chequear_particion_detecta_la_fuga_de_informacion() -> None:
    """Entrenar con fuga es peor que no entrenar: produce metricas en las que se confia."""
    features = _features(200)
    X = features.drop(columns=[TARGET])
    y = features[TARGET]

    particion = Particion(X.iloc[:150], X.iloc[100:], y.iloc[:150], y.iloc[100:])

    with pytest.raises(ErrorValidacion, match="indices_solapados"):
        chequear_particion(particion)


def test_ejecutar_pipeline_guarda_el_informe_de_los_chequeos(
    parquet_features: Path, tmp_path: Path
) -> None:
    """Los resultados de las validaciones de la particion se persisten con las metricas."""
    ruta_chequeos = tmp_path / "chequeos.parquet"

    resultado = ejecutar_pipeline(
        RutasEntrenamiento(
            parquet_features,
            tmp_path / "modelo.joblib",
            tmp_path / "metricas.parquet",
            ruta_chequeos,
        ),
        nombre_modelo="ridge",
        semilla=SEMILLA,
        con_referencias=False,
    )

    assert ruta_chequeos.exists()
    guardado = pd.read_parquet(ruta_chequeos)
    assert guardado.equals(resultado.chequeos_particion)
    assert not guardado.empty


def test_en_modo_estricto_una_particion_sospechosa_detiene_el_entrenamiento(
    parquet_features: Path, tmp_path: Path
) -> None:
    """Con 20 filas de prueba salta la advertencia de tamano; en estricto, no se entrena."""
    ruta_modelo = tmp_path / "modelo.joblib"

    with pytest.raises(ErrorValidacion, match="tamano_del_conjunto_de_prueba"):
        ejecutar_pipeline(
            RutasEntrenamiento(parquet_features, ruta_modelo, tmp_path / "metricas.parquet"),
            nombre_modelo="ridge",
            semilla=SEMILLA,
            particion_estricta=True,
        )

    assert not ruta_modelo.exists()


def test_main_devuelve_codigo_de_error_si_la_particion_no_pasa_los_chequeos(
    parquet_features: Path, tmp_path: Path
) -> None:
    """El error es controlado: informe, codigo 1 y ningun artefacto escrito."""
    ruta_modelo = tmp_path / "modelo.joblib"

    codigo = main(
        [
            "--features",
            str(parquet_features),
            "--modelo",
            "ridge",
            "--modelo-salida",
            str(ruta_modelo),
            "--metricas",
            str(tmp_path / "metricas.parquet"),
            "--chequeos-particion",
            str(tmp_path / "chequeos.parquet"),
            "--particion-estricta",
        ]
    )

    assert codigo == 1
    assert not ruta_modelo.exists()


def test_una_advertencia_no_detiene_el_entrenamiento_por_defecto(
    parquet_features: Path, tmp_path: Path
) -> None:
    """La deriva se avisa y se registra; solo la fuga bloquea el pipeline."""
    resultado = ejecutar_pipeline(
        RutasEntrenamiento(
            parquet_features,
            tmp_path / "modelo.joblib",
            tmp_path / "metricas.parquet",
            tmp_path / "chequeos.parquet",
        ),
        nombre_modelo="ridge",
        semilla=SEMILLA,
        con_referencias=False,
    )

    assert resultado.chequeos_particion is not None
    assert "advertencia" in set(resultado.chequeos_particion["severidad"])
