import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from data.preprocesamiento import construir_preprocesamiento
from data.validacion import ErrorValidacion
from pipelines.feature_pipeline.feature_pipeline import TARGET
from pipelines.inference_pipeline.inference_pipeline import (
    COLUMNAS_SALIDA,
    cargar_modelo,
    ejecutar_pipeline,
    guardar_predicciones,
    leer_datos_nuevos,
    main,
    mostrar_predicciones,
    predecir_lote,
    preparar_datos_nuevos,
    resumir_lote,
    separar_contexto,
)

FILAS = 60
SEMILLA = 3
PREDICCION_CONSTANTE = 0.75
PREDICCION_BAJA = 0.30
TOTAL_PORCENTAJE = 100.0
# el archivo de aspirantes de las pruebas tiene dos filas
ASPIRANTES = 2
FILAS_PARQUET = 5
CESTAS = 3

CABECERA = "GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research"
FILAS_CSV = "325,113,4,4.5,n/a,9.1,1\n300,100,2,3,3,7.8,0\n"


def _datos_entrenamiento(filas: int = FILAS) -> tuple[pd.DataFrame, pd.Series]:
    """Perfiles sintéticos con los que ajustar los modelos de las pruebas."""
    generador = np.random.default_rng(SEMILLA)
    X = pd.DataFrame(
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
    y = pd.Series(0.3 + 0.06 * (X["cgpa"] - 6.8), name=TARGET)
    return X, y


def _modelo_dummy(constante: float = PREDICCION_CONSTANTE) -> Pipeline:
    """Pipeline con el mismo preprocesamiento del entrenamiento y un modelo dummy.

    Es el modelo que pide el enunciado: predice siempre lo mismo, así que cualquier cifra
    que aparezca en la salida viene del pipeline y no del azar de un ajuste.
    """
    X, y = _datos_entrenamiento()
    modelo = Pipeline(
        [
            ("preparacion", construir_preprocesamiento()),
            ("modelo", DummyRegressor(strategy="constant", constant=constante)),
        ]
    )
    return modelo.fit(X, y)


def _modelo_lineal() -> Pipeline:
    """Pipeline real ajustado, para comprobar que el preprocesamiento se aplica."""
    X, y = _datos_entrenamiento()
    modelo = Pipeline([("preparacion", construir_preprocesamiento()), ("modelo", Ridge(alpha=1.0))])
    return modelo.fit(X, y)


@pytest.fixture
def modelo_guardado(tmp_path: Path) -> Path:
    """Artefacto en disco, como el que deja el training pipeline."""
    ruta = tmp_path / "modelos" / "modelo_dummy.joblib"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_modelo_dummy(), ruta)
    return ruta


@pytest.fixture
def csv_aspirantes(tmp_path: Path) -> Path:
    """Archivo de aspirantes nuevos: sin objetivo, con un dato ausente y con un `id`."""
    ruta = tmp_path / "aspirantes.csv"
    ruta.write_text(f"id,{CABECERA}\nA-1,325,113,4,4.5,n/a,9.1,1\nA-2,300,100,2,3,3,7.8,0\n")
    return ruta


# --- Carga del modelo -------------------------------------------------------------------


def test_cargar_modelo_devuelve_el_pipeline_entrenado(modelo_guardado: Path) -> None:
    """La evidencia de carga: el artefacto vuelve con su preprocesamiento y su estimador."""
    modelo = cargar_modelo(modelo_guardado)

    assert list(modelo.named_steps) == ["preparacion", "modelo"]
    assert isinstance(modelo.named_steps["modelo"], DummyRegressor)


def test_cargar_modelo_falla_con_un_mensaje_util(tmp_path: Path) -> None:
    """El mensaje dice qué ejecutar antes, que es lo que necesita quien lo lee."""
    with pytest.raises(FileNotFoundError, match="train_pipeline"):
        cargar_modelo(tmp_path / "no_existe.joblib")


# --- Lectura de los datos nuevos --------------------------------------------------------


def test_leer_datos_nuevos_admite_csv(csv_aspirantes: Path) -> None:
    """El CSV se lee como texto: los tipos se deciden después, explícitamente."""
    datos = leer_datos_nuevos(csv_aspirantes)

    assert len(datos) == ASPIRANTES
    assert datos["LOR "].iloc[0] == "n/a"


def test_leer_datos_nuevos_admite_parquet(tmp_path: Path) -> None:
    """Un lote generado por otro sistema suele llegar en parquet."""
    ruta = tmp_path / "aspirantes.parquet"
    _datos_entrenamiento(5)[0].to_parquet(ruta, index=False)

    assert len(leer_datos_nuevos(ruta)) == FILAS_PARQUET


def test_leer_datos_nuevos_rechaza_un_formato_desconocido(tmp_path: Path) -> None:
    """Mejor un mensaje con las opciones que un error de pandas tres líneas más abajo."""
    ruta = tmp_path / "aspirantes.xlsx"
    ruta.write_text("no importa")

    with pytest.raises(ValueError, match="Formato no soportado"):
        leer_datos_nuevos(ruta)


def test_leer_datos_nuevos_falla_si_no_existe_el_archivo(tmp_path: Path) -> None:
    """El caso más común en producción: la ruta del lote está mal."""
    with pytest.raises(FileNotFoundError, match="datos nuevos"):
        leer_datos_nuevos(tmp_path / "no_existe.csv")


# --- Separacion de predictores y contexto -----------------------------------------------


def test_separar_contexto_normaliza_nombres_y_aparta_las_columnas_extra(
    csv_aspirantes: Path,
) -> None:
    """El `id` no entra al modelo, pero tiene que volver a salir: si no, ¿de quién es cada
    predicción?"""
    predictores, contexto = separar_contexto(leer_datos_nuevos(csv_aspirantes))

    assert "lor" in predictores.columns
    assert list(contexto.columns) == ["id"]
    assert contexto["id"].tolist() == ["A-1", "A-2"]


def test_separar_contexto_falla_si_faltan_predictores() -> None:
    """Sin las columnas que el modelo espera no hay nada que predecir."""
    with pytest.raises(ValueError, match="faltan columnas"):
        separar_contexto(pd.DataFrame({"GRE Score": ["325"]}))


# --- Preparacion de los datos -----------------------------------------------------------


def test_preparar_datos_nuevos_funciona_sin_la_columna_objetivo(csv_aspirantes: Path) -> None:
    """El caso de inferencia por definición: el aspirante aún no ha sido admitido."""
    predictores, _ = separar_contexto(leer_datos_nuevos(csv_aspirantes))

    preparados = preparar_datos_nuevos(predictores)

    assert TARGET not in preparados.columns
    assert str(preparados["gre_score"].dtype) == "Int64"
    assert str(preparados["research"].dtype) == "boolean"
    assert preparados["lor"].isna().sum() == 1  # el 'n/a' llego como ausente


def test_preparar_datos_nuevos_rechaza_un_valor_imposible(tmp_path: Path) -> None:
    """Un GRE de 900 detiene el lote antes de generar una prediccion que nadie deberia usar."""
    ruta = tmp_path / "malos.csv"
    ruta.write_text(f"{CABECERA}\n900,118,4,4.5,4.5,9.65,1\n")
    predictores, _ = separar_contexto(leer_datos_nuevos(ruta))

    with pytest.raises(ErrorValidacion, match="fuera_de_rango"):
        preparar_datos_nuevos(predictores)


def test_preparar_datos_nuevos_rechaza_una_categoria_desconocida(tmp_path: Path) -> None:
    """Un rating 9 no puede convertirse en nulo en silencio."""
    ruta = tmp_path / "malos.csv"
    ruta.write_text(f"{CABECERA}\n325,118,9,4.5,4.5,9.65,1\n")
    predictores, _ = separar_contexto(leer_datos_nuevos(ruta))

    with pytest.raises(ErrorValidacion, match="categoria_invalida"):
        preparar_datos_nuevos(predictores)


def test_preparar_datos_nuevos_rechaza_un_perfil_casi_vacio(tmp_path: Path) -> None:
    """Con menos de cuatro predictores conocidos no hay perfil que puntuar."""
    ruta = tmp_path / "vacios.csv"
    ruta.write_text(f"{CABECERA}\nn/a,n/a,n/a,n/a,3,8.1,n/a\n")
    predictores, _ = separar_contexto(leer_datos_nuevos(ruta))

    with pytest.raises(ErrorValidacion, match="registro_utilizable"):
        preparar_datos_nuevos(predictores)


def test_los_perfiles_repetidos_se_admiten_en_inferencia(tmp_path: Path) -> None:
    """Dos aspirantes pueden tener el mismo perfil y ambos merecen su prediccion."""
    ruta = tmp_path / "repetidos.csv"
    ruta.write_text(f"{CABECERA}\n{FILAS_CSV.splitlines()[1]}\n{FILAS_CSV.splitlines()[1]}\n")
    predictores, _ = separar_contexto(leer_datos_nuevos(ruta))

    assert len(preparar_datos_nuevos(predictores)) == ASPIRANTES


# --- Generacion de predicciones ---------------------------------------------------------


def test_predecir_lote_devuelve_prediccion_cesta_e_intervalo(csv_aspirantes: Path) -> None:
    """La salida habla el lenguaje del producto, no solo el del modelo."""
    predictores = preparar_datos_nuevos(separar_contexto(leer_datos_nuevos(csv_aspirantes))[0])

    predicciones = predecir_lote(_modelo_dummy(), predictores)

    assert list(predicciones.columns) == COLUMNAS_SALIDA
    assert predicciones["prediccion"].tolist() == [PREDICCION_CONSTANTE] * 2
    assert set(predicciones["cesta"]) == {"probable"}
    assert predicciones["limite_inferior"].lt(predicciones["prediccion"]).all()
    assert predicciones["limite_superior"].gt(predicciones["prediccion"]).all()
    assert not predicciones["advertencia"].any()


def test_un_perfil_debil_llega_marcado_con_la_advertencia(csv_aspirantes: Path) -> None:
    """Bajo el umbral, el modelo tiende a ser optimista y el lote debe decirlo."""
    predictores = preparar_datos_nuevos(separar_contexto(leer_datos_nuevos(csv_aspirantes))[0])

    predicciones = predecir_lote(_modelo_dummy(PREDICCION_BAJA), predictores)

    assert predicciones["advertencia"].all()
    assert set(predicciones["cesta"]) == {"ambiciosa"}


def test_las_predicciones_se_acotan_al_rango_valido(csv_aspirantes: Path) -> None:
    """Una probabilidad de 1.5 no existe: el lote no puede publicarla."""
    predictores = preparar_datos_nuevos(separar_contexto(leer_datos_nuevos(csv_aspirantes))[0])

    predicciones = predecir_lote(_modelo_dummy(1.5), predictores)

    assert predicciones["prediccion"].tolist() == [1.0, 1.0]


def test_se_aplican_las_transformaciones_aprendidas_en_entrenamiento() -> None:
    """La evidencia de que el dato ausente se imputa con la mediana **de entrenamiento**.

    Un aspirante sin CGPA recibe la misma prediccion que otro idéntico cuyo CGPA fuera la
    mediana que el pipeline aprendió al ajustarse. Si el script transformara por su cuenta,
    esta igualdad se rompería.
    """
    modelo = _modelo_lineal()
    X_entrenamiento, _ = _datos_entrenamiento()
    mediana = float(X_entrenamiento["cgpa"].median())
    fila = X_entrenamiento.head(1).copy()

    sin_cgpa = predecir_lote(modelo, fila.assign(cgpa=np.nan))
    con_mediana = predecir_lote(modelo, fila.assign(cgpa=mediana))

    assert sin_cgpa["prediccion"].iloc[0] == pytest.approx(con_mediana["prediccion"].iloc[0])


# --- Resumen y almacenamiento -----------------------------------------------------------


def test_resumir_lote_reparte_las_predicciones_por_cesta() -> None:
    """El resumen que se mira primero: si todo sale 'segura', algo cambio."""
    predicciones = pd.DataFrame(
        {
            "prediccion": [0.5, 0.72, 0.95],
            "cesta": ["ambiciosa", "probable", "segura"],
        }
    )

    resumen = resumir_lote(predicciones)

    assert resumen["n"].sum() == CESTAS
    assert resumen["porcentaje"].sum() == pytest.approx(TOTAL_PORCENTAJE)
    assert resumen["cesta"].tolist() == ["ambiciosa", "probable", "segura"]


def test_resumir_un_lote_vacio_no_rompe() -> None:
    """Un archivo sin filas es un caso raro, pero no debe tumbar el pipeline."""
    assert resumir_lote(pd.DataFrame(columns=["prediccion", "cesta"])).empty


@pytest.mark.parametrize("extension", [".parquet", ".csv"])
def test_guardar_predicciones_escribe_un_archivo_relegible(tmp_path: Path, extension: str) -> None:
    """El CSV existe porque estas predicciones las consume gente, no solo programas."""
    predicciones = pd.DataFrame({"prediccion": [0.8], "cesta": ["segura"]})
    ruta = tmp_path / "salidas" / f"predicciones{extension}"

    guardar_predicciones(predicciones, ruta)

    leido = pd.read_parquet(ruta) if extension == ".parquet" else pd.read_csv(ruta)
    assert leido["prediccion"].tolist() == [0.8]


def test_guardar_predicciones_rechaza_un_formato_desconocido(tmp_path: Path) -> None:
    """Fallar al guardar es peor que fallar al empezar: mejor decirlo claro."""
    with pytest.raises(ValueError, match="Formato no soportado"):
        guardar_predicciones(pd.DataFrame({"prediccion": [0.8]}), tmp_path / "salida.txt")


def test_mostrar_predicciones_solo_escribe_si_se_pide(caplog: pytest.LogCaptureFixture) -> None:
    """La visualizacion es opcional: por defecto el log no se llena de filas."""
    predicciones = pd.DataFrame({"prediccion": [0.8], "cesta": ["segura"]})

    with caplog.at_level(logging.INFO):
        mostrar_predicciones(predicciones, 0)
        assert "Primeras" not in caplog.text

        mostrar_predicciones(predicciones, 1)
        assert "Primeras 1 predicciones" in caplog.text


# --- Ejecucion completa -----------------------------------------------------------------


def test_ejecutar_pipeline_de_extremo_a_extremo(
    modelo_guardado: Path, csv_aspirantes: Path, tmp_path: Path
) -> None:
    """Modelo en disco + archivo de aspirantes -> predicciones en disco."""
    salida = tmp_path / "predicciones.parquet"

    resultado = ejecutar_pipeline(modelo_guardado, csv_aspirantes, salida)

    assert salida.exists()
    guardado = pd.read_parquet(salida)
    assert guardado["id"].tolist() == ["A-1", "A-2"]  # el contexto vuelve a salir
    assert all(columna in guardado.columns for columna in COLUMNAS_SALIDA)
    assert guardado["prediccion"].tolist() == [PREDICCION_CONSTANTE] * 2
    assert resultado.ruta_salida == salida
    assert not resultado.resumen.empty


def test_ejecutar_pipeline_puede_no_guardar_nada(
    modelo_guardado: Path, csv_aspirantes: Path
) -> None:
    """Para inspeccionar un lote sin dejar rastro en `07_model_output`."""
    resultado = ejecutar_pipeline(modelo_guardado, csv_aspirantes, None)

    assert resultado.ruta_salida is None
    assert len(resultado.predicciones) == ASPIRANTES


def test_el_objetivo_no_entra_al_modelo_si_viene_en_el_archivo(
    modelo_guardado: Path, tmp_path: Path
) -> None:
    """Repredecir un historico es legitimo; usar la respuesta para predecirla, no."""
    ruta = tmp_path / "historico.csv"
    ruta.write_text(f"{CABECERA},Chance of Admit\n325,113,4,4.5,4.5,9.1,1,0.92\n")

    resultado = ejecutar_pipeline(modelo_guardado, ruta, None)

    assert resultado.predicciones["prediccion"].iloc[0] == PREDICCION_CONSTANTE
    assert resultado.predicciones[TARGET].iloc[0] == "0.92"


def test_main_ejecuta_el_script_desde_la_linea_de_comandos(
    modelo_guardado: Path, csv_aspirantes: Path, tmp_path: Path
) -> None:
    """La forma en que se ejecuta de forma autonoma."""
    salida = tmp_path / "predicciones.csv"

    codigo = main(
        [
            "--modelo",
            str(modelo_guardado),
            "--datos",
            str(csv_aspirantes),
            "--salida",
            str(salida),
            "--mostrar",
            "2",
        ]
    )

    assert codigo == 0
    assert len(pd.read_csv(salida)) == ASPIRANTES


def test_main_no_guarda_nada_con_sin_guardar(
    modelo_guardado: Path, csv_aspirantes: Path, tmp_path: Path
) -> None:
    """La opcion para probar un lote sin escribir el archivo de salida."""
    salida = tmp_path / "predicciones.parquet"

    codigo = main(
        [
            "--modelo",
            str(modelo_guardado),
            "--datos",
            str(csv_aspirantes),
            "--salida",
            str(salida),
            "--sin-guardar",
        ]
    )

    assert codigo == 0
    assert not salida.exists()


def test_main_devuelve_codigo_de_error_si_falta_el_modelo(
    csv_aspirantes: Path, tmp_path: Path
) -> None:
    """Un orquestador necesita un codigo de salida, no un traceback."""
    salida = tmp_path / "predicciones.parquet"

    codigo = main(
        [
            "--modelo",
            str(tmp_path / "no_existe.joblib"),
            "--datos",
            str(csv_aspirantes),
            "--salida",
            str(salida),
        ]
    )

    assert codigo == 1
    assert not salida.exists()


def test_main_devuelve_codigo_de_error_con_datos_invalidos(
    modelo_guardado: Path, tmp_path: Path
) -> None:
    """Un dato imposible detiene el lote sin escribir predicciones que nadie deberia usar."""
    entrada = tmp_path / "malos.csv"
    entrada.write_text(f"{CABECERA}\n900,118,4,4.5,4.5,9.65,1\n")
    salida = tmp_path / "predicciones.parquet"

    codigo = main(
        ["--modelo", str(modelo_guardado), "--datos", str(entrada), "--salida", str(salida)]
    )

    assert codigo == 1
    assert not salida.exists()
