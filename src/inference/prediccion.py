"""Lógica de inferencia de la demo: cargar el modelo, predecir y explicar.

Se mantiene separada de la interfaz de Streamlit para poder probarla con `pytest`, que
es donde se detectan los errores de verdad: una interfaz se ve rota, una regla de negocio
mal escrita no.

Las constantes de esta capa salen de los notebooks, no de la intuición:

- `CORTES_CESTA` son los terciles del objetivo aprendidos en entrenamiento (`03.3`).
- `MAE_MODELO` y `PREDICCION_MINIMA` se midieron en `06` y `07`.
- `UMBRAL_ADVERTENCIA` viene de la recomendación de `07`: por debajo de 0.55 el modelo
  sobrestima y su error se duplica, así que la interfaz debe avisarlo.
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

MAE_MODELO = 0.0476
PREDICCION_MINIMA = 0.447
UMBRAL_ADVERTENCIA = 0.55

NOMBRES_LEGIBLES: dict[str, str] = {
    "gre_score": "Puntaje GRE",
    "toefl_score": "Puntaje TOEFL",
    "university_rating": "Calificación de la universidad",
    "sop": "Carta de intención (SOP)",
    "lor": "Cartas de recomendación (LOR)",
    "cgpa": "Promedio acumulado (CGPA)",
    "research": "Experiencia en investigación",
}


def ruta_modelo_por_defecto() -> Path:
    """Ruta del artefacto entrenado, resuelta desde la raíz del repositorio."""
    actual = Path(__file__).resolve()
    for candidato in actual.parents:
        if (candidato / "pyproject.toml").exists():
            return candidato / "data" / "06_models" / "modelo_final_automl.joblib"
    raise FileNotFoundError("No se encontro la raiz del proyecto")


def cargar_modelo(ruta: Path | None = None) -> Pipeline:
    """Carga el pipeline de preprocesamiento + modelo entrenado en `06-models`."""
    destino = ruta or ruta_modelo_por_defecto()
    if not destino.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {destino}. "
            "Ejecuta el notebook 06.Seleccion-de-modelo-AutoML para generarlo."
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
