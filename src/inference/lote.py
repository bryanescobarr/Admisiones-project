"""Procesamiento por lotes para la demo: de un archivo subido a predicciones descargables.

La pestaña de lotes de la demo necesita tres cosas que la interfaz no debería resolver por
su cuenta: leer un archivo que llega como bytes, puntuarlo y devolverlo listo para
descargar. Todo eso vive aquí para poder probarlo con `pytest`, que es donde se detectan los
errores de verdad: una interfaz rota se ve, una tabla mal construida no.

**No hay una segunda implementación de la inferencia.** La predicción la hace
`predecir_dataframe` del inference pipeline, el mismo código que ejecuta el script de línea
de comandos. Este módulo solo adapta la entrada y la salida al navegador.
"""

import sys
from io import BytesIO
from pathlib import Path
from typing import IO

import pandas as pd
from sklearn.pipeline import Pipeline

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipelines.inference_pipeline.inference_pipeline import (  # noqa: E402
    COLUMNAS_SALIDA,
    EXTENSIONES_TABULARES,
    predecir_dataframe,
    resumir_lote,
)

# archivos de ejemplo versionados en el repositorio, para que cualquiera pueda probar la
# pestana de lotes sin tener que fabricarse un archivo antes
DIR_EJEMPLOS = _SRC.parent / "ejemplos"
ARCHIVO_EJEMPLO_ENTRADA = DIR_EJEMPLOS / "aspirantes_ejemplo.csv"
ARCHIVO_EJEMPLO_SALIDA = DIR_EJEMPLOS / "predicciones_ejemplo.csv"

# limite de filas por lote: la demo corre en un contenedor pequeno de Streamlit Cloud y un
# archivo enorme la dejaria sin memoria sin decir por que
MAX_FILAS_LOTE = 5000


def leer_tabla_subida(archivo: IO[bytes], nombre: str) -> pd.DataFrame:
    """Lee el archivo que subió la persona, en CSV o parquet.

    El CSV se lee como texto (`dtype="object"`, sin interpretar nulos) igual que en el
    pipeline: quien decide qué es un nulo y de qué tipo es cada columna es el contrato de
    entrada, no la heurística de pandas.
    """
    extension = Path(nombre).suffix.lower()
    if extension == ".csv":
        datos = pd.read_csv(archivo, dtype="object", na_filter=False)
    elif extension == ".parquet":
        datos = pd.read_parquet(archivo)
    else:
        raise ValueError(
            f"Formato no soportado: '{extension}'. Sube un archivo {list(EXTENSIONES_TABULARES)}"
        )
    if datos.empty:
        raise ValueError("El archivo no tiene ninguna fila")
    if len(datos) > MAX_FILAS_LOTE:
        raise ValueError(
            f"El archivo tiene {len(datos)} filas y el limite de la demo son "
            f"{MAX_FILAS_LOTE}. Para lotes mayores usa el script del inference pipeline."
        )
    return datos


def procesar_lote(modelo: Pipeline, datos: pd.DataFrame) -> pd.DataFrame:
    """Puntúa el lote reutilizando el inference pipeline, sin tocar ninguna transformación."""
    return predecir_dataframe(modelo, datos)


def resumen_del_lote(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Distribución por cesta, que es lo primero que mira quien recibe un lote."""
    return resumir_lote(predicciones)


def columnas_de_prediccion(predicciones: pd.DataFrame) -> list[str]:
    """Columnas añadidas por el modelo, para poder destacarlas en la tabla de la demo."""
    return [columna for columna in COLUMNAS_SALIDA if columna in predicciones.columns]


def a_csv(predicciones: pd.DataFrame) -> bytes:
    """Convierte las predicciones en el CSV que se descarga desde el navegador."""
    contenido = BytesIO()
    predicciones.to_csv(contenido, index=False)
    return contenido.getvalue()


def ejemplo_de_entrada() -> bytes:
    """Contenido del archivo de ejemplo, para ofrecerlo como descarga en la demo."""
    if not ARCHIVO_EJEMPLO_ENTRADA.exists():
        raise FileNotFoundError(f"Falta el archivo de ejemplo en {ARCHIVO_EJEMPLO_ENTRADA}")
    return ARCHIVO_EJEMPLO_ENTRADA.read_bytes()
