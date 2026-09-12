from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from data.preprocesamiento import construir_preprocesamiento
from data.validacion import ErrorValidacion
from inference.lote import (
    ARCHIVO_EJEMPLO_ENTRADA,
    ARCHIVO_EJEMPLO_SALIDA,
    MAX_FILAS_LOTE,
    a_csv,
    columnas_de_prediccion,
    ejemplo_de_entrada,
    leer_tabla_subida,
    procesar_lote,
    resumen_del_lote,
)

PREDICCION_CONSTANTE = 0.75
ASPIRANTES_EJEMPLO = 8
SEMILLA = 5
FILAS_ENTRENAMIENTO = 40
TOTAL_PORCENTAJE = 100.0
FILAS_DEL_LOTE = 2

CABECERA = "id,GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research"
FILA_VALIDA = "A-1,325,113,4,4.5,n/a,9.1,1"


def _datos_entrenamiento() -> tuple[pd.DataFrame, pd.Series]:
    """Perfiles sintéticos con los que ajustar el modelo dummy de las pruebas."""
    generador = np.random.default_rng(SEMILLA)
    X = pd.DataFrame(
        {
            "gre_score": generador.integers(290, 341, FILAS_ENTRENAMIENTO).astype("float64"),
            "toefl_score": generador.integers(95, 121, FILAS_ENTRENAMIENTO).astype("float64"),
            "university_rating": generador.integers(1, 6, FILAS_ENTRENAMIENTO).astype("float64"),
            "sop": generador.integers(2, 11, FILAS_ENTRENAMIENTO) / 2,
            "lor": generador.integers(2, 11, FILAS_ENTRENAMIENTO) / 2,
            "cgpa": generador.uniform(6.8, 9.9, FILAS_ENTRENAMIENTO).round(2),
            "research": generador.integers(0, 2, FILAS_ENTRENAMIENTO).astype("float64"),
        }
    )
    return X, pd.Series(0.3 + 0.06 * (X["cgpa"] - 6.8), name="chance_of_admit")


def _modelo_dummy() -> Pipeline:
    """Pipeline con el preprocesamiento real y un modelo que predice siempre lo mismo."""
    X, y = _datos_entrenamiento()
    modelo = Pipeline(
        [
            ("preparacion", construir_preprocesamiento()),
            ("modelo", DummyRegressor(strategy="constant", constant=PREDICCION_CONSTANTE)),
        ]
    )
    return modelo.fit(X, y)


def _csv(filas: str) -> BytesIO:
    """Archivo CSV en memoria, como el que llega del navegador."""
    return BytesIO(f"{CABECERA}\n{filas}".encode())


# --- Lectura del archivo subido ---------------------------------------------------------


def test_lee_un_csv_subido() -> None:
    """El CSV llega como bytes y se lee como texto, sin que pandas interprete nada."""
    datos = leer_tabla_subida(_csv(f"{FILA_VALIDA}\n"), "aspirantes.csv")

    assert len(datos) == 1
    assert datos["LOR "].iloc[0] == "n/a"


def test_lee_un_parquet_subido(tmp_path: Path) -> None:
    """Un lote generado por otro sistema suele llegar en parquet."""
    ruta = tmp_path / "aspirantes.parquet"
    _datos_entrenamiento()[0].to_parquet(ruta, index=False)

    with ruta.open("rb") as archivo:
        assert len(leer_tabla_subida(archivo, ruta.name)) == FILAS_ENTRENAMIENTO


def test_rechaza_un_formato_que_no_sabe_leer() -> None:
    """El mensaje dice qué formatos valen, que es lo que necesita quien lo lee."""
    with pytest.raises(ValueError, match="Formato no soportado"):
        leer_tabla_subida(BytesIO(b"cualquier cosa"), "aspirantes.xlsx")


def test_rechaza_un_archivo_vacio() -> None:
    """Subir la cabecera sola es un error tan común como silencioso."""
    with pytest.raises(ValueError, match="ninguna fila"):
        leer_tabla_subida(_csv(""), "aspirantes.csv")


def test_rechaza_un_lote_demasiado_grande() -> None:
    """La demo corre en un contenedor pequeño: mejor un mensaje que quedarse sin memoria."""
    filas = "\n".join([FILA_VALIDA] * (MAX_FILAS_LOTE + 1))

    with pytest.raises(ValueError, match="limite de la demo"):
        leer_tabla_subida(_csv(filas), "aspirantes.csv")


# --- Procesamiento del lote -------------------------------------------------------------


def test_procesar_lote_anade_las_columnas_de_prediccion() -> None:
    """Cada fila sale con su probabilidad, su cesta, su rango y su advertencia."""
    datos = leer_tabla_subida(_csv(f"{FILA_VALIDA}\n"), "aspirantes.csv")

    predicciones = procesar_lote(_modelo_dummy(), datos)

    assert predicciones["prediccion"].tolist() == [PREDICCION_CONSTANTE]
    assert columnas_de_prediccion(predicciones) == [
        "prediccion",
        "cesta",
        "limite_inferior",
        "limite_superior",
        "advertencia",
    ]


def test_procesar_lote_conserva_las_columnas_de_contexto() -> None:
    """Sin el `id` de vuelta, nadie sabe de quién es cada predicción."""
    datos = leer_tabla_subida(_csv(f"{FILA_VALIDA}\n"), "aspirantes.csv")

    predicciones = procesar_lote(_modelo_dummy(), datos)

    assert predicciones["id"].tolist() == ["A-1"]


def test_procesar_lote_rechaza_un_dato_imposible() -> None:
    """Un GRE de 900 detiene el lote entero: es preferible a publicar una predicción mala."""
    datos = leer_tabla_subida(_csv("A-1,900,113,4,4.5,4.5,9.1,1\n"), "aspirantes.csv")

    with pytest.raises(ErrorValidacion, match="fuera_de_rango"):
        procesar_lote(_modelo_dummy(), datos)


def test_resumen_del_lote_reparte_por_cesta() -> None:
    """El resumen que se mira primero: cómo se distribuyen las opciones."""
    datos = leer_tabla_subida(_csv(f"{FILA_VALIDA}\n{FILA_VALIDA}\n"), "aspirantes.csv")

    resumen = resumen_del_lote(procesar_lote(_modelo_dummy(), datos))

    assert resumen["n"].sum() == FILAS_DEL_LOTE
    assert resumen["porcentaje"].sum() == pytest.approx(TOTAL_PORCENTAJE)


# --- Descarga ---------------------------------------------------------------------------


def test_a_csv_devuelve_un_archivo_relegible() -> None:
    """Lo que se descarga tiene que poder abrirse en una hoja de cálculo."""
    predicciones = pd.DataFrame({"id": ["A-1"], "prediccion": [0.8], "cesta": ["segura"]})

    contenido = a_csv(predicciones)

    assert pd.read_csv(BytesIO(contenido))["cesta"].tolist() == ["segura"]


# --- Archivos de ejemplo versionados ----------------------------------------------------


def test_el_archivo_de_ejemplo_esta_en_el_repositorio() -> None:
    """La demo lo ofrece como descarga: si falta, esa parte de la interfaz se rompe."""
    contenido = ejemplo_de_entrada()

    assert ARCHIVO_EJEMPLO_ENTRADA.exists()
    assert contenido.startswith(b"id,nombre,GRE Score")


def test_el_ejemplo_de_entrada_se_puede_procesar() -> None:
    """La evidencia de que el archivo de ejemplo sirve para lo que dice servir."""
    datos = leer_tabla_subida(BytesIO(ejemplo_de_entrada()), ARCHIVO_EJEMPLO_ENTRADA.name)

    predicciones = procesar_lote(_modelo_dummy(), datos)

    assert len(predicciones) == ASPIRANTES_EJEMPLO
    assert predicciones["nombre"].notna().all()


def test_el_ejemplo_de_salida_corresponde_al_de_entrada() -> None:
    """Entrada y salida versionadas tienen que contar la misma historia."""
    entrada = pd.read_csv(ARCHIVO_EJEMPLO_ENTRADA)
    salida = pd.read_csv(ARCHIVO_EJEMPLO_SALIDA)

    assert len(entrada) == len(salida) == ASPIRANTES_EJEMPLO
    assert salida["id"].tolist() == entrada["id"].tolist()
    assert salida["prediccion"].between(0, 1).all()
    assert set(salida["cesta"]) <= {"segura", "probable", "ambiciosa"}
