"""Inference pipeline del proyecto de admisiones: del modelo entrenado a predicciones.

Tercera y última etapa de la arquitectura FTI de `src/README.md`. Carga el modelo que dejó
el training pipeline, lee un archivo con aspirantes nuevos, les aplica **exactamente** las
mismas transformaciones del entrenamiento y guarda las predicciones.

    data/06_models/modelo_training_pipeline.joblib + un archivo de aspirantes
        -> data/07_model_output/predicciones.parquet

**Por qué aquí no se reimplementa ninguna transformación.** El artefacto `.joblib` no
guarda solo el estimador: guarda el `Pipeline` completo, con el paso de preparación y los
parámetros que aprendió en entrenamiento —la mediana de cada columna, la media y la
desviación del escalado, el mapeo del encoder—. Llamar a `modelo.predict()` aplica esa
misma secuencia. Cualquier transformación que este script hiciera por su cuenta sería una
*segunda* versión de la verdad, y el día que las dos se separen las predicciones
empezarían a estar mal sin que nadie lo note.

Lo que sí ocurre antes es la **puesta en formato**: un archivo de aspirantes nuevos llega
con los nombres del CSV original (`'LOR '`), con `'n/a'` donde falta un dato y con todo
como texto. Eso se normaliza y se valida contra el contrato de entrada, que es el mismo
dominio documentado del feature pipeline pero **sin la columna objetivo**: un aspirante que
todavía no ha sido admitido no la tiene, y exigirla sería absurdo.

**La predicción no es solo un número.** Cada fila sale con su cesta (`segura`, `probable`,
`ambiciosa`), su rango de referencia de ±1 MAE y la advertencia del tramo bajo, usando la
misma lógica que la demo (`src/inference/prediccion.py`): lo que ve quien consume el lote
tiene que coincidir con lo que ve quien usa el formulario.

Uso:

    uv run python src/pipelines/inference_pipeline/inference_pipeline.py
    uv run python src/pipelines/inference_pipeline/inference_pipeline.py --datos aspirantes.csv
    uv run python src/pipelines/inference_pipeline/inference_pipeline.py --mostrar 10
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# el script se ejecuta directamente, asi que `src/` no esta en sys.path todavia
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.validacion import (  # noqa: E402
    ErrorValidacion,
    EsquemaDatos,
    ReglaColumna,
    validar,
)
from inference.prediccion import (  # noqa: E402
    MAE_MODELO,
    UMBRAL_ADVERTENCIA,
    clasificar_cesta,
    intervalo_estimado,
)
from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    CATEGORIAS_VALIDAS_RATING,
    COLS_PREDICTORAS,
    RANGOS,
    RELACION_REGISTRO_UTILIZABLE,
    TARGET,
    a_snake_case,
    buscar_raiz_proyecto,
    tipar_columnas,
    unificar_nulos,
)

logger = logging.getLogger(__name__)

RUTA_MODELO_RELATIVA = Path("data") / "06_models" / "modelo_training_pipeline.joblib"
RUTA_DATOS_RELATIVA = Path("data") / "01_raw" / "Admission_Predict.csv"
RUTA_SALIDA_RELATIVA = Path("data") / "07_model_output" / "predicciones.parquet"

# columnas que anade este pipeline a cada fila de entrada
COLUMNAS_SALIDA: list[str] = [
    "prediccion",
    "cesta",
    "limite_inferior",
    "limite_superior",
    "advertencia",
]

EXTENSIONES_TABULARES = (".csv", ".parquet")

# Contrato de entrada: el dominio documentado del feature pipeline, sin el objetivo.
#
# Dos diferencias deliberadas respecto al contrato de las features:
#
# 1. **No se exige `chance_of_admit`**: es justo lo que se va a predecir.
# 2. **Se permiten filas repetidas**: dos aspirantes distintos pueden tener el mismo perfil
#    y ambos merecen su prediccion. En entrenamiento un duplicado sesga el ajuste; en
#    inferencia es un caso normal.
ESQUEMA_ENTRADA = EsquemaDatos(
    columnas={
        "gre_score": ReglaColumna(tipos=("Int64",), rango=RANGOS["gre_score"]),
        "toefl_score": ReglaColumna(tipos=("Int64",), rango=RANGOS["toefl_score"]),
        "university_rating": ReglaColumna(
            tipos=("category",), categorias=CATEGORIAS_VALIDAS_RATING
        ),
        "sop": ReglaColumna(tipos=("Float64",), rango=RANGOS["sop"]),
        "lor": ReglaColumna(tipos=("Float64",), rango=RANGOS["lor"]),
        "cgpa": ReglaColumna(tipos=("Float64",), rango=RANGOS["cgpa"]),
        "research": ReglaColumna(tipos=("boolean",), categorias=(0.0, 1.0)),
    },
    filas_unicas=False,
    relaciones=(RELACION_REGISTRO_UTILIZABLE,),
)


@dataclass(frozen=True)
class ResultadoInferencia:
    """Lo que produce una ejecución completa del inference pipeline."""

    predicciones: pd.DataFrame
    resumen: pd.DataFrame
    ruta_salida: Path | None = None


def cargar_modelo(ruta: Path) -> Pipeline:
    """Carga el modelo entrenado y deja constancia de qué se cargó.

    El registro no es decorativo: cuando una predicción sorprende, la primera pregunta es
    siempre «¿qué modelo estaba sirviendo?», y la respuesta tiene que estar en el log de la
    ejecución que la produjo.
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {ruta}. Ejecuta antes el training pipeline: "
            "uv run python src/pipelines/training_pipeline/train_pipeline.py"
        )
    modelo = joblib.load(ruta)
    pasos = list(getattr(modelo, "named_steps", {}))
    estimador = modelo.named_steps["modelo"] if "modelo" in pasos else modelo
    logger.info(
        "Modelo cargado de %s (%.1f KB): pasos %s, estimador %s",
        ruta,
        ruta.stat().st_size / 1024,
        pasos or "sin pipeline",
        type(estimador).__name__,
    )
    return modelo


def leer_datos_nuevos(ruta: Path) -> pd.DataFrame:
    """Lee el archivo de aspirantes, en CSV o parquet.

    El CSV se lee como texto, sin que pandas decida tipos ni nulos: las dos decisiones se
    toman después, explícitamente, igual que en el feature pipeline.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos nuevos en {ruta}")
    if ruta.suffix == ".parquet":
        datos = pd.read_parquet(ruta)
    elif ruta.suffix == ".csv":
        datos = pd.read_csv(ruta, dtype="object", na_filter=False)
    else:
        raise ValueError(
            f"Formato no soportado: '{ruta.suffix}'. Usa uno de {list(EXTENSIONES_TABULARES)}"
        )
    logger.info("Leidas %d filas x %d columnas de %s", len(datos), datos.shape[1], ruta)
    return datos


def separar_contexto(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el archivo en predictores y columnas de contexto.

    Un archivo real trae más cosas que los siete predictores: un identificador, el nombre de
    la universidad a la que se postula, la fecha de la consulta. Esas columnas no entran al
    modelo —fallaría, porque no las vio al entrenar— pero **tienen que volver a salir** en el
    archivo de predicciones, o nadie sabrá de quién es cada número.
    """
    renombrado = datos.rename(columns=a_snake_case)
    faltantes = [columna for columna in COLS_PREDICTORAS if columna not in renombrado.columns]
    if faltantes:
        raise ValueError(f"Al archivo de aspirantes le faltan columnas: {faltantes}")
    contexto = renombrado.drop(columns=COLS_PREDICTORAS)
    return renombrado[COLS_PREDICTORAS], contexto


def preparar_datos_nuevos(datos: pd.DataFrame) -> pd.DataFrame:
    """Pone los datos en el formato que espera el modelo y valida que sean utilizables.

    Reutiliza las funciones del feature pipeline —`unificar_nulos` y `tipar_columnas`— para
    que un `'n/a'` o un `'9'` en `university_rating` signifiquen lo mismo aquí que al
    entrenar. Después valida contra `ESQUEMA_ENTRADA`: un GRE de 900 o un perfil casi vacío
    detienen el lote antes de generar una predicción que nadie debería usar.
    """
    predictores = unificar_nulos(datos)
    predictores = tipar_columnas(predictores)
    validar(predictores, ESQUEMA_ENTRADA, contexto="los datos de inferencia")
    logger.info(
        "Datos preparados: %d filas, %d valores ausentes que el modelo imputara",
        len(predictores),
        int(predictores.isna().sum().sum()),
    )
    return predictores


def predecir_lote(modelo: Pipeline, predictores: pd.DataFrame) -> pd.DataFrame:
    """Predice y traduce cada probabilidad al lenguaje del producto.

    `modelo.predict()` aplica el preprocesamiento aprendido en entrenamiento —imputación,
    recorte al dominio, codificación y escalado—, así que aquí no se toca ningún valor. Lo
    que se añade es la interpretación: la cesta, el rango de ±1 MAE y la advertencia del
    tramo en el que el modelo tiende a ser optimista, con la misma lógica que la demo.
    """
    crudas = np.asarray(modelo.predict(predictores), dtype=float)
    probabilidades = np.clip(crudas, 0.0, 1.0)
    intervalos = [intervalo_estimado(float(valor)) for valor in probabilidades]
    resultado = pd.DataFrame(
        {
            "prediccion": probabilidades,
            "cesta": [clasificar_cesta(float(valor)) for valor in probabilidades],
            "limite_inferior": [inferior for inferior, _ in intervalos],
            "limite_superior": [superior for _, superior in intervalos],
            "advertencia": probabilidades < UMBRAL_ADVERTENCIA,
        },
        index=predictores.index,
    )
    logger.info(
        "Predicciones generadas: %d filas, media %.3f, rango [%.3f, %.3f], +-%.4f de MAE",
        len(resultado),
        float(resultado["prediccion"].mean()),
        float(resultado["prediccion"].min()),
        float(resultado["prediccion"].max()),
        MAE_MODELO,
    )
    return resultado


def predecir_dataframe(modelo: Pipeline, datos: pd.DataFrame) -> pd.DataFrame:
    """Puntúa una tabla de aspirantes ya cargada en memoria.

    Es el recorrido completo de una fila —separar el contexto, poner en formato, validar y
    predecir— sin pasar por el disco, de modo que **el script de línea de comandos y la
    pestaña de lotes de la demo ejecutan exactamente el mismo código**. Si hubiera dos
    caminos, tarde o temprano darían números distintos para el mismo archivo.

    Devuelve la tabla de entrada con las columnas de predicción añadidas: primero el
    contexto (identificadores y demás), después los predictores ya tipados y al final la
    predicción con su interpretación.
    """
    predictores, contexto = separar_contexto(datos)
    predictores = preparar_datos_nuevos(predictores)
    predicciones = predecir_lote(modelo, predictores)
    return pd.concat([contexto, predictores, predicciones], axis=1)


def resumir_lote(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Distribución de las predicciones por cesta: el resumen que se mira primero.

    Si un lote sale con el 90 % de aspirantes en «segura», lo raro no es el lote: es que el
    modelo o los datos han cambiado, y conviene mirarlo antes de enviar nada.
    """
    if predicciones.empty:
        return pd.DataFrame(columns=["cesta", "n", "porcentaje", "prediccion_media"])
    resumen = (
        predicciones.groupby("cesta", observed=True)
        .agg(n=("prediccion", "size"), prediccion_media=("prediccion", "mean"))
        .reset_index()
    )
    resumen["porcentaje"] = 100 * resumen["n"] / len(predicciones)
    return resumen[["cesta", "n", "porcentaje", "prediccion_media"]].sort_values("prediccion_media")


def guardar_predicciones(predicciones: pd.DataFrame, ruta: Path) -> Path:
    """Escribe las predicciones en la capa `07_model_output`, en parquet o CSV.

    El CSV existe porque estas predicciones las consume gente, no solo programas: quien
    orienta a los aspirantes abre una hoja de cálculo, no un parquet.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.suffix == ".csv":
        predicciones.to_csv(ruta, index=False)
    elif ruta.suffix == ".parquet":
        predicciones.to_parquet(ruta, index=False, engine="pyarrow", compression="snappy")
    else:
        raise ValueError(
            f"Formato no soportado: '{ruta.suffix}'. Usa uno de {list(EXTENSIONES_TABULARES)}"
        )
    logger.info("Predicciones guardadas en %s (%.1f KB)", ruta, ruta.stat().st_size / 1024)
    return ruta


def mostrar_predicciones(predicciones: pd.DataFrame, filas: int = 0) -> None:
    """Escribe en el log las primeras predicciones, para inspeccionarlas sin abrir el archivo."""
    if filas <= 0 or predicciones.empty:
        return
    columnas = [columna for columna in COLUMNAS_SALIDA if columna in predicciones.columns]
    logger.info(
        "Primeras %d predicciones:\n%s",
        min(filas, len(predicciones)),
        predicciones.head(filas)[columnas].round(4).to_string(index=False),
    )


def ejecutar_pipeline(
    ruta_modelo: Path,
    ruta_datos: Path,
    ruta_salida: Path | None = None,
    *,
    filas_a_mostrar: int = 0,
) -> ResultadoInferencia:
    """Ejecuta la inferencia completa: cargar, preparar, predecir y guardar.

    Si el archivo de entrada trae la columna objetivo —por ejemplo, porque se está
    reprediciendo un histórico para comprobar el modelo—, se usa **solo para comparar**, se
    registra el error del lote y nunca se le pasa al modelo. Ese error no es una métrica de
    rendimiento: si el archivo incluye filas que el modelo vio al entrenar, saldrá
    optimista. Las métricas honestas están en el training pipeline, medidas sobre un
    conjunto de prueba que el modelo no vio.
    """
    modelo = cargar_modelo(ruta_modelo)
    datos = leer_datos_nuevos(ruta_datos)
    salida = predecir_dataframe(modelo, datos)

    if TARGET in salida.columns:
        objetivo = pd.to_numeric(salida[TARGET], errors="coerce")
        comparables = objetivo.notna()
        if comparables.any():
            error = float(
                (salida.loc[comparables, "prediccion"] - objetivo[comparables]).abs().mean()
            )
            logger.info(
                "El archivo traia el objetivo: MAE del lote %.4f sobre %d filas (solo "
                "comparacion, no entro al modelo; si el archivo incluye filas que el modelo "
                "vio al entrenar, este error sera optimista)",
                error,
                int(comparables.sum()),
            )

    resumen = resumir_lote(salida)
    logger.info("Distribucion por cesta:\n%s", resumen.round(3).to_string(index=False))
    mostrar_predicciones(salida, filas_a_mostrar)

    ruta_guardada = guardar_predicciones(salida, ruta_salida) if ruta_salida else None
    return ResultadoInferencia(salida, resumen, ruta_guardada)


def _parsear_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    """Define la interfaz de línea de comandos del script."""
    raiz = buscar_raiz_proyecto()
    parser = argparse.ArgumentParser(
        description="Inference pipeline: aspirantes nuevos -> predicciones de admision."
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=raiz / RUTA_MODELO_RELATIVA,
        help="Artefacto entrenado a cargar (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=raiz / RUTA_DATOS_RELATIVA,
        help=(
            "CSV o parquet con los aspirantes a puntuar. Por defecto usa el archivo crudo "
            "del proyecto como si fueran aspirantes nuevos (por defecto: %(default)s)"
        ),
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=raiz / RUTA_SALIDA_RELATIVA,
        help="Archivo de predicciones a generar, .parquet o .csv (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--mostrar",
        type=int,
        default=0,
        help="Escribe en el log las primeras N predicciones (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--sin-guardar",
        action="store_true",
        help="Solo muestra las predicciones, sin escribir ningun archivo",
    )
    return parser.parse_args(argumentos)


def main(argumentos: list[str] | None = None) -> int:
    """Punto de entrada del script. Devuelve 0 si la inferencia terminó bien, 1 si no."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    opciones = _parsear_argumentos(argumentos)
    try:
        resultado = ejecutar_pipeline(
            opciones.modelo,
            opciones.datos,
            None if opciones.sin_guardar else opciones.salida,
            filas_a_mostrar=opciones.mostrar,
        )
    except (ErrorValidacion, FileNotFoundError, ValueError) as error:
        # un modelo que falta, un archivo mal formado o un dato invalido no son fallos del
        # programa: se reportan como informe y codigo de salida, no como traceback
        informe = str(error)
    else:
        logger.info("Inference pipeline terminado: %d predicciones", len(resultado.predicciones))
        return 0
    logger.error("%s", informe)
    logger.error("No se generaron predicciones: no se escribio ningun archivo")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
