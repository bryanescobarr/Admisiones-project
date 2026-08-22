"""Modelo heurístico de admisión como estimador de scikit-learn.

Formaliza la regla manual propuesta en `notebooks/3-analysis/03.3`: normalizar cada
variable a [0, 1] según su dominio documentado y promediarlas sin ponderar. Envolverla en
la interfaz de scikit-learn permite compararla con `DummyRegressor` y con cualquier
modelo usando las mismas particiones y las mismas métricas.

A diferencia del resto de modelos, **consume los datos crudos, no la salida del pipeline
de preprocesamiento**: la normalización por dominio no tiene sentido sobre variables ya
estandarizadas. Esa es también su ventaja, porque no necesita imputación: si a un
aspirante le falta un dato, la regla promedia los que sí tiene.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from data.transformaciones import a_tipos_sklearn

# (minimo, maximo) usados para llevar cada columna a [0, 1]
DOMINIOS_NORMALIZACION: dict[str, tuple[float, float]] = {
    "gre_score": (260, 340),
    "toefl_score": (0, 120),
    "university_rating": (1, 5),
    "sop": (1, 5),
    "lor": (1, 5),
    "cgpa": (0, 10),
    "research": (0, 1),
}


class HeuristicaAdmision(BaseEstimator, RegressorMixin):
    """Promedio de las variables normalizadas por su dominio documentado.

    Parameters
    ----------
    recalibrar:
        Si es `True`, ajusta en `fit` una recta `a + b * puntaje` que lleva el promedio a
        la escala del objetivo. Sin recalibrar, el puntaje se usa tal cual como
        predicción: ordena igual de bien, pero está desplazado respecto al objetivo.
    """

    def __init__(self, recalibrar: bool = True) -> None:
        self.recalibrar = recalibrar

    def _puntuar(self, datos: pd.DataFrame) -> np.ndarray:
        """Promedio de las columnas conocidas, normalizadas a [0, 1], ignorando nulos."""
        numericos = a_tipos_sklearn(pd.DataFrame(datos))
        normalizadas = []
        for columna, (minimo, maximo) in DOMINIOS_NORMALIZACION.items():
            if columna in numericos.columns:
                normalizadas.append((numericos[columna] - minimo) / (maximo - minimo))
        if not normalizadas:
            raise ValueError("Ninguna columna conocida por la heuristica esta presente")
        return np.nanmean(np.column_stack(normalizadas), axis=1)

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "HeuristicaAdmision":
        """Aprende únicamente los dos coeficientes de la recalibración."""
        puntaje = self._puntuar(X)
        if self.recalibrar:
            if y is None:
                raise ValueError("Se necesita 'y' para recalibrar la heuristica")
            diseno = np.c_[np.ones(len(puntaje)), puntaje]
            coeficientes, *_ = np.linalg.lstsq(diseno, np.asarray(y, dtype=float), rcond=None)
            self.coef_ = coeficientes
        else:
            self.coef_ = np.array([0.0, 1.0])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice la probabilidad de admisión, acotada al rango válido [0, 1]."""
        check_is_fitted(self, "coef_")
        puntaje = self._puntuar(X)
        return np.clip(self.coef_[0] + self.coef_[1] * puntaje, 0.0, 1.0)
