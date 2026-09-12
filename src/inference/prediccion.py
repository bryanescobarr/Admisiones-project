"""Lógica de inferencia de la demo: cargar el modelo, predecir y explicar.

Se mantiene separada de la interfaz de Streamlit para poder probarla con `pytest`, que
es donde se detectan los errores de verdad: una interfaz se ve rota, una regla de negocio
mal escrita no.

**Qué modelo se sirve.** El artefacto por defecto es `modelo_produccion.joblib`, el que
produce `src/pipelines/training_pipeline/train_pipeline.py`. Antes se servía el `.joblib`
exportado a mano desde el notebook `06`; ahora la demo sirve lo que genera la cadena
reproducible —feature pipeline, chequeos de partición, entrenamiento y validación— y no una
pieza suelta. Los artefactos del POC siguen versionados como referencia histórica.

Las constantes de esta capa no se inventan:

- `CORTES_CESTA` son los terciles del objetivo aprendidos en entrenamiento (`03.3`). No
  dependen del modelo: son una propiedad de los datos.
- `MAE_MODELO` se **lee del informe del training pipeline**
  (`data/08_reporting/metricas_training_pipeline.parquet`), de la fila del modelo servido,
  para que la cifra que ve el usuario sea la que midió la ejecución que generó el artefacto.
- `PREDICCION_MINIMA` se resuelve por el mismo camino, con una salvedad: ese informe **no
  trae hoy una columna con la predicción mínima**, y añadirla exigiría tocar el training
  pipeline, que este cambio no modifica. Se usa entonces el valor medido sobre el artefacto
  versionado, y `tests/test_prediccion.py` lo recalcula desde los datos crudos para que no
  pueda quedarse describiendo a otro modelo. El día que el informe incluya esa columna, se
  leerá sola.
- `UMBRAL_ADVERTENCIA` marca el tramo con menos ejemplos de entrenamiento y más error.

**Por qué hay valores de respaldo.** `data/**` está en `.gitignore` y este cambio solo
versiona el artefacto de servicio, así que en Streamlit Cloud el informe de métricas no
existe. Si falta, se usan las cifras medidas en la ejecución que produjo el `.joblib`
versionado: la demo nunca debe caerse por no encontrar un archivo de reportes.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

# columnas que espera el modelo, en el orden en que se entrenó
COLUMNAS_ENTRADA: list[str] = [
    "gre_score",
    "toefl_score",
    "university_rating",
    "sop",
    "lor",
    "cgpa",
    "research",
]

# (minimo, maximo) admitidos por el formulario, del dominio documentado en Informacion.txt
LIMITES_ENTRADA: dict[str, tuple[float, float]] = {
    "gre_score": (260, 340),
    "toefl_score": (0, 120),
    "university_rating": (1, 5),
    "sop": (1, 5),
    "lor": (1, 5),
    "cgpa": (0, 10),
}

CORTES_CESTA: tuple[float, float] = (0.67, 0.79)
ETIQUETAS_CESTA: tuple[str, str, str] = ("ambiciosa", "probable", "segura")

# artefacto de servicio, generado por el training pipeline
MODELO_SERVIDO = "modelo_produccion.joblib"

# informe del training pipeline y fila que corresponde al modelo servido
RUTA_METRICAS_RELATIVA = Path("data") / "08_reporting" / "metricas_training_pipeline.parquet"
MODELO_EN_METRICAS = "extra_trees"

# cifras medidas en la ejecucion que genero el artefacto versionado, sobre las 118 filas de
# prueba que no entraron al ajuste. Sirven de respaldo cuando el informe no esta disponible
MAE_DE_RESPALDO = 0.0467
PREDICCION_MINIMA_DE_RESPALDO = 0.4066

NOMBRES_LEGIBLES: dict[str, str] = {
    "gre_score": "Puntaje GRE",
    "toefl_score": "Puntaje TOEFL",
    "university_rating": "Calificación de la universidad",
    "sop": "Carta de intención (SOP)",
    "lor": "Cartas de recomendación (LOR)",
    "cgpa": "Promedio acumulado (CGPA)",
    "research": "Experiencia en investigación",
}


def raiz_proyecto() -> Path:
    """Raíz del repositorio, resuelta subiendo hasta encontrar el `pyproject.toml`."""
    actual = Path(__file__).resolve()
    for candidato in actual.parents:
        if (candidato / "pyproject.toml").exists():
            return candidato
    raise FileNotFoundError("No se encontro la raiz del proyecto")


def ruta_modelo_por_defecto() -> Path:
    """Ruta del artefacto de servicio, resuelta desde la raíz del repositorio."""
    return raiz_proyecto() / "data" / "06_models" / MODELO_SERVIDO


def ruta_metricas_por_defecto() -> Path:
    """Ruta del informe de métricas que escribe el training pipeline."""
    return raiz_proyecto() / RUTA_METRICAS_RELATIVA


def metrica_del_modelo_servido(columna: str, por_defecto: float) -> float:
    """Devuelve una métrica del modelo servido leída del informe del training pipeline.

    Si el informe no está —el caso normal en el despliegue, porque `data/**` no se
    versiona—, o si no trae esa columna, se usa el valor medido en la ejecución que generó
    el artefacto. Una demo no puede quedarse sin arrancar porque falte un archivo de
    reportes, y tampoco debe inventarse una cifra: por eso el respaldo está documentado y
    hay una prueba que lo recalcula.
    """
    ruta = ruta_metricas_por_defecto()
    if not ruta.exists():
        return float(por_defecto)
    metricas = pd.read_parquet(ruta)
    fila = metricas.loc[metricas["modelo"] == MODELO_EN_METRICAS]
    if fila.empty or columna not in fila.columns or pd.isna(fila.iloc[0][columna]):
        return float(por_defecto)
    return float(fila.iloc[0][columna])


# error medio del modelo servido y prediccion mas baja que llego a emitir en prueba
MAE_MODELO: float = metrica_del_modelo_servido("MAE", MAE_DE_RESPALDO)
PREDICCION_MINIMA: float = metrica_del_modelo_servido(
    "prediccion_minima", PREDICCION_MINIMA_DE_RESPALDO
)

# por debajo de este valor hay muy pocos ejemplos de entrenamiento y el error medido sube un
# 20 % (0.0552 frente a 0.0460): la interfaz debe avisarlo
UMBRAL_ADVERTENCIA = 0.55


def cargar_modelo(ruta: Path | None = None) -> Pipeline:
    """Carga el pipeline de preprocesamiento + modelo entrenado en `06-models`."""
    destino = ruta or ruta_modelo_por_defecto()
    if not destino.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {destino}. Generalo con el training pipeline: "
            "uv run python src/pipelines/training_pipeline/train_pipeline.py "
            f"--modelo-salida data/06_models/{MODELO_SERVIDO}"
        )
    return joblib.load(destino)


def construir_fila(datos: dict[str, Any]) -> pd.DataFrame:
    """Convierte las respuestas del formulario en la fila que espera el pipeline.

    Los valores ausentes se dejan como `np.nan`: el pipeline los imputa con la mediana
    aprendida en entrenamiento, así que un aspirante puede consultar sin tener todos los
    datos a mano.
    """
    fila = {columna: datos.get(columna, np.nan) for columna in COLUMNAS_ENTRADA}
    return pd.DataFrame([fila], columns=COLUMNAS_ENTRADA)


def predecir(modelo: Pipeline, datos: dict[str, Any]) -> float:
    """Devuelve la probabilidad de admisión estimada, acotada al rango válido."""
    prediccion = float(modelo.predict(construir_fila(datos))[0])
    return float(np.clip(prediccion, 0.0, 1.0))


def clasificar_cesta(probabilidad: float) -> str:
    """Traduce la probabilidad a la cesta que usa el producto."""
    corte_bajo, corte_alto = CORTES_CESTA
    if probabilidad < corte_bajo:
        return ETIQUETAS_CESTA[0]
    if probabilidad < corte_alto:
        return ETIQUETAS_CESTA[1]
    return ETIQUETAS_CESTA[2]


def intervalo_estimado(probabilidad: float) -> tuple[float, float]:
    """Rango de referencia de ±1 MAE alrededor de la predicción.

    No es un intervalo de confianza estadístico —calcularlo bien es el experimento 2 del
    plan de `07`— sino una forma honesta de comunicar que el número tiene un margen.
    """
    return (
        float(np.clip(probabilidad - MAE_MODELO, 0.0, 1.0)),
        float(np.clip(probabilidad + MAE_MODELO, 0.0, 1.0)),
    )


def contribuciones(modelo: Pipeline, datos: dict[str, Any]) -> pd.Series:
    """Aporte de cada atributo a esta predicción concreta, según SHAP.

    Devuelve una serie ordenada por magnitud: cuánto sumó o restó cada dato respecto a la
    predicción media del modelo.
    """
    preprocesador = modelo.named_steps["preparacion"]
    estimador = modelo.named_steps["modelo"]
    fila_preparada = preprocesador.transform(construir_fila(datos))

    explicador = shap.TreeExplainer(estimador)
    valores = np.ravel(explicador.shap_values(fila_preparada))
    serie = pd.Series(valores, index=list(fila_preparada.columns))
    return serie.reindex(serie.abs().sort_values(ascending=False).index)


def prediccion_media(modelo: Pipeline) -> float:
    """Predicción media del modelo (valor base de SHAP): el punto de partida."""
    estimador = modelo.named_steps["modelo"]
    return float(np.ravel(shap.TreeExplainer(estimador).expected_value)[0])
