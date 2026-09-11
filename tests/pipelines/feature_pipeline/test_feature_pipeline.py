from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import (
    COLUMNAS_DERIVADAS,
    COLUMNAS_ESPERADAS,
    TARGET,
    a_snake_case,
    buscar_raiz_proyecto,
    construir_features,
    ejecutar_pipeline,
    eliminar_duplicados,
    guardar_features,
    leer_datos_crudos,
    main,
    normalizar_nombres,
    tipar_columnas,
    unificar_nulos,
    validar_dominio,
)

# filas distintas que quedan del CSV de ejemplo tras eliminar la copia exacta
FILAS_UNICAS = 2

# una fila valida, una con centinelas de nulo ('n/a', '-') y una copia exacta de la primera
CSV_EJEMPLO = """GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research,Chance of Admit
337,118,4,4.5,4.5,9.65,1,0.92
n/a,107,3,4,-,8.87,0,0.76
337,118,4,4.5,4.5,9.65,1,0.92
"""


@pytest.fixture
def csv_ejemplo(tmp_path: Path) -> Path:
    """CSV crudo mínimo con las mismas rarezas que la fuente real."""
    ruta = tmp_path / "Admission_Predict.csv"
    ruta.write_text(CSV_EJEMPLO, encoding="utf-8")
    return ruta


def _datos_tipados() -> pd.DataFrame:
    """Dataset ya limpio y tipado, tal y como sale de `tipar_columnas`."""
    return pd.DataFrame(
        {
            "gre_score": pd.array([337, None], dtype="Int64"),
            "toefl_score": pd.array([118, 107], dtype="Int64"),
            "university_rating": pd.Categorical(
                ["4", "3"], categories=["1", "2", "3", "4", "5"], ordered=True
            ),
            "sop": pd.array([4.5, 4.0], dtype="Float64"),
            "lor": pd.array([4.5, None], dtype="Float64"),
            "cgpa": pd.array([9.65, 8.87], dtype="Float64"),
            "research": pd.array([True, False], dtype="boolean"),
            TARGET: pd.array([0.92, 0.76], dtype="Float64"),
        }
    )


def test_a_snake_case_limpia_los_nombres_del_csv() -> None:
    """'LOR ' con espacio final y 'GRE Score' con mayúsculas quedan normalizados."""
    assert a_snake_case("LOR ") == "lor"
    assert a_snake_case("GRE Score") == "gre_score"


def test_leer_datos_crudos_no_interpreta_tipos_ni_nulos(csv_ejemplo: Path) -> None:
    """Todo llega como texto: ni pandas decide los tipos ni convierte 'n/a' en NaN."""
    crudos = leer_datos_crudos(csv_ejemplo)

    assert (crudos.dtypes == "object").all()
    assert crudos.loc[1, "GRE Score"] == "n/a"
    assert crudos.shape == (3, 8)


def test_leer_datos_crudos_falla_si_no_existe_el_archivo(tmp_path: Path) -> None:
    """Un error claro es mejor que un stack trace de pandas tres líneas más abajo."""
    with pytest.raises(FileNotFoundError, match="datos crudos"):
        leer_datos_crudos(tmp_path / "no_existe.csv")


def test_normalizar_nombres_deja_las_columnas_esperadas(csv_ejemplo: Path) -> None:
    """Los ocho nombres quedan en snake_case y en el orden documentado."""
    normalizado = normalizar_nombres(leer_datos_crudos(csv_ejemplo))

    assert list(normalizado.columns) == COLUMNAS_ESPERADAS


def test_normalizar_nombres_falla_si_falta_una_columna() -> None:
    """Si la fuente cambia, el pipeline se rompe aquí y no genera un parquet incompleto."""
    with pytest.raises(ValueError, match="Faltan columnas esperadas"):
        normalizar_nombres(pd.DataFrame({"GRE Score": ["337"]}))


def test_unificar_nulos_convierte_los_centinelas_en_na() -> None:
    """'n/a', '-' y la celda vacía significan lo mismo: dato ausente."""
    datos = pd.DataFrame({"cgpa": ["9.65", "n/a", "-", "", "  8.0  "]})

    unificado = unificar_nulos(datos)

    assert unificado["cgpa"].isna().tolist() == [False, True, True, True, False]
    assert unificado.loc[4, "cgpa"] == "8.0"


def test_eliminar_duplicados_descarta_las_filas_identicas(csv_ejemplo: Path) -> None:
    """La tercera fila del ejemplo es copia exacta de la primera."""
    datos = unificar_nulos(normalizar_nombres(leer_datos_crudos(csv_ejemplo)))

    assert len(eliminar_duplicados(datos)) == FILAS_UNICAS


def test_tipar_columnas_asigna_los_tipos_nullable() -> None:
    """Cada columna recibe su tipo: Int64, Float64, categórica ordinal y boolean."""
    datos = unificar_nulos(
        pd.DataFrame(
            {
                "gre_score": ["337"],
                "toefl_score": ["118"],
                "university_rating": ["4"],
                "sop": ["4.5"],
                "lor": ["n/a"],
                "cgpa": ["9.65"],
                "research": ["1"],
                TARGET: ["0.92"],
            }
        )
    )

    tipado = tipar_columnas(datos)

    assert str(tipado["gre_score"].dtype) == "Int64"
    assert str(tipado["cgpa"].dtype) == "Float64"
    assert str(tipado["research"].dtype) == "boolean"
    assert tipado["university_rating"].dtype.ordered
    assert tipado["research"].iloc[0]
    assert tipado["lor"].isna().all()


def test_tipar_columnas_falla_con_un_research_no_booleano() -> None:
    """`research` solo admite 0 y 1: cualquier otro valor es un error de la fuente."""
    datos = pd.DataFrame(
        {
            "gre_score": ["337"],
            "toefl_score": ["118"],
            "university_rating": ["4"],
            "sop": ["4.5"],
            "lor": ["4.5"],
            "cgpa": ["9.65"],
            "research": ["2"],
            TARGET: ["0.92"],
        }
    )

    with pytest.raises(ValueError, match="no booleanos"):
        tipar_columnas(datos)


def test_tipar_columnas_falla_si_una_conversion_descarta_valores() -> None:
    """Un texto no numérico no puede desaparecer en silencio convertido en nulo."""
    datos = pd.DataFrame(
        {
            "gre_score": ["trescientos"],
            "toefl_score": ["118"],
            "university_rating": ["4"],
            "sop": ["4.5"],
            "lor": ["4.5"],
            "cgpa": ["9.65"],
            "research": ["1"],
            TARGET: ["0.92"],
        }
    )

    with pytest.raises(ValueError, match="descartados al convertir"):
        tipar_columnas(datos)


def test_validar_dominio_acepta_los_datos_correctos() -> None:
    """El dataset de ejemplo respeta los rangos de Informacion.txt."""
    datos = _datos_tipados()

    assert validar_dominio(datos) is datos


def test_validar_dominio_detecta_un_valor_imposible() -> None:
    """Un GRE de 900 en la fuente es un problema de captura, no algo que recortar."""
    datos = _datos_tipados()
    datos.loc[0, "gre_score"] = 900

    with pytest.raises(ValueError, match="fuera del dominio"):
        validar_dominio(datos)


def test_validar_dominio_rechaza_un_objetivo_nulo() -> None:
    """Una fila sin `chance_of_admit` no sirve para entrenar."""
    datos = _datos_tipados()
    datos.loc[0, TARGET] = pd.NA

    with pytest.raises(ValueError, match="no puede tener nulos"):
        validar_dominio(datos)


def test_construir_features_devuelve_float64_con_el_objetivo_al_final() -> None:
    """scikit-learn necesita float64 y np.nan, no los dtypes nullable de pandas."""
    features = construir_features(_datos_tipados())

    assert list(features.columns) == COLUMNAS_ESPERADAS
    assert (features.dtypes == "float64").all()
    assert features.loc[0].tolist() == [337.0, 118.0, 4.0, 4.5, 4.5, 9.65, 1.0, 0.92]
    assert bool(features.loc[1, "gre_score"] != features.loc[1, "gre_score"])  # np.nan


def test_construir_features_no_imputa_los_nulos() -> None:
    """La imputación se aprende en entrenamiento: hacerla aquí filtraría información."""
    features = construir_features(_datos_tipados())

    assert features["gre_score"].isna().sum() == 1
    assert features["lor"].isna().sum() == 1


def test_construir_features_con_derivados_anade_los_tres_atributos() -> None:
    """Con la opción activada aparecen los atributos propuestos en 4-feat_eng."""
    features = construir_features(_datos_tipados(), con_derivados=True)

    assert all(columna in features.columns for columna in COLUMNAS_DERIVADAS)
    assert features.loc[0, "sop_lor_media"] == pytest.approx(4.5)
    assert features.loc[0, "rating_x_research"] == pytest.approx(4.0)
    assert list(features.columns)[-1] == TARGET


def test_guardar_features_escribe_un_parquet_relegible(tmp_path: Path) -> None:
    """El artefacto solo sirve si se relee con la misma forma y los mismos valores."""
    features = construir_features(_datos_tipados())
    ruta = tmp_path / "subdirectorio" / "features.parquet"

    guardar_features(features, ruta)

    assert ruta.exists()
    assert pd.read_parquet(ruta).equals(features)


def test_ejecutar_pipeline_de_extremo_a_extremo(csv_ejemplo: Path, tmp_path: Path) -> None:
    """Del CSV crudo al parquet de features, con duplicado y centinelas por el camino."""
    salida = tmp_path / "features.parquet"

    features = ejecutar_pipeline(csv_ejemplo, salida)

    assert len(features) == FILAS_UNICAS  # la fila duplicada se eliminó
    assert features["gre_score"].isna().sum() == 1  # el 'n/a' llegó como nulo
    assert features["lor"].isna().sum() == 1  # el '-' también
    assert pd.read_parquet(salida).equals(features)


def test_main_ejecuta_el_script_con_argumentos(csv_ejemplo: Path, tmp_path: Path) -> None:
    """La interfaz de línea de comandos es la forma en que se ejecuta de forma autónoma."""
    salida = tmp_path / "features_cli.parquet"

    codigo = main(["--entrada", str(csv_ejemplo), "--salida", str(salida), "--con-derivados"])

    assert codigo == 0
    guardadas = pd.read_parquet(salida)
    assert all(columna in guardadas.columns for columna in COLUMNAS_DERIVADAS)


def test_las_rutas_por_defecto_apuntan_a_la_fuente_del_proyecto() -> None:
    """Sin argumentos, el script lee el CSV real del repositorio."""
    raiz = buscar_raiz_proyecto()

    assert (raiz / "pyproject.toml").exists()
    assert (raiz / "data" / "01_raw" / "Admission_Predict.csv").exists()


def test_el_pipeline_reproduce_el_dataset_de_los_notebooks(tmp_path: Path) -> None:
    """Sobre la fuente real produce lo mismo que la cadena de notebooks 02 y 04.

    Es la prueba que evita el peor fallo posible de esta etapa: que el script automatizado
    y los cuadernos, partiendo del mismo CSV, dejen de coincidir sin que nadie se entere.
    """
    raiz = buscar_raiz_proyecto()
    intermedio = raiz / "data" / "02_intermediate" / "admisiones_type_fixed.parquet"
    if not intermedio.exists():
        pytest.skip("Falta el parquet de 02_intermediate generado por el notebook 02")

    esperado = pd.read_parquet(intermedio)
    obtenido = ejecutar_pipeline(
        raiz / "data" / "01_raw" / "Admission_Predict.csv",
        tmp_path / "admisiones_features.parquet",
    )

    assert len(obtenido) == len(esperado)
    assert obtenido[TARGET].to_numpy().tolist() == esperado[TARGET].astype(float).tolist()
    assert obtenido["cgpa"].isna().sum() == int(esperado["cgpa"].isna().sum())
