"""Validación del modelo: ¿generaliza, o solo funciona sobre los datos que ya vio?

Una métrica en el conjunto de prueba es **un número con una sola muestra**: cambia la
semilla de la partición y cambia el número. Este módulo la acompaña de lo que hace falta
para decidir si el modelo sirve, siguiendo la [guía de validación del modelo][guia]:

1. **Validación cruzada sobre entrenamiento**, con la variante adecuada al problema
   (`kfold`, `repetido`, `estratificado` por cuantiles del objetivo o `series_temporales`).
2. **Las mismas métricas en los tres sitios** —entrenamiento, validación cruzada y
   prueba— en una sola tabla, que es donde se ven las diferencias.
3. **Un diagnóstico explícito** de subajuste, sobreajuste, inestabilidad y degradación,
   cada uno con la acción recomendada. Un informe que dice «MAE 0.047» no ayuda a decidir;
   uno que dice «sobreajuste: el modelo memoriza, limita la profundidad» sí.
4. **Curva de aprendizaje** y **segmentos débiles**, las dos evidencias que responden a
   «¿esto se arregla con más datos?» y «¿a quién le funciona peor?».

[guia]: https://joserzapata.github.io/courses/ciencia-datos-en-produccion/model-validation/

**Por qué no `deepchecks`.** La guía lo usa para esto mismo. El criterio, igual que en
`src/data/validacion.py` y `src/data/particion.py`, fue no añadir una dependencia para algo
que `scikit-learn` ya sabe hacer; los umbrales del diagnóstico (degradación relativa de
0.1) son los que usa su suite, así que los veredictos coinciden.
"""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    learning_curve,
)

logger = logging.getLogger(__name__)

# --- Umbrales del diagnostico, todos documentados donde se usan -------------------------

# proporcion del error de validacion que puede explicarse por la brecha con entrenamiento
# antes de hablar de sobreajuste: por encima, el modelo acierta sobre todo donde ya miro
BRECHA_SOBREAJUSTE = 0.50
BRECHA_SOBREAJUSTE_LEVE = 0.20

# R2 de validacion por debajo del cual el modelo no esta capturando la senal disponible
R2_SUBAJUSTE = 0.50

# desviacion entre pliegues, como proporcion del error medio, por encima de la cual el
# resultado depende demasiado de que filas cayeron en cada pliegue
INESTABILIDAD = 0.25

# degradacion relativa admitida entre validacion cruzada y prueba (umbral de deepchecks)
DEGRADACION_MAXIMA = 0.10

# un segmento cuyo error supera este multiplo del error global es un segmento debil
FACTOR_SEGMENTO_DEBIL = 1.5

# una columna con menos valores distintos que esto no se puede partir en segmentos
MIN_VALORES_PARA_SEGMENTAR = 2

TALLAS_CURVA = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Correlación de rangos, nula cuando el modelo predice siempre lo mismo."""
    if float(np.max(y_pred) - np.min(y_pred)) == 0.0:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).statistic)


# metricas de regresion: MAE y RMSE para el error, R2 y MAPE para comunicar y spearman
# porque el producto ordena universidades. Las `neg_*` de scikit-learn se reconvierten a
# positivo mas abajo, para que la tabla se lea como se habla.
METRICAS_REGRESION: dict[str, str | object] = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
    "MAPE": "neg_mean_absolute_percentage_error",
    "spearman": make_scorer(_spearman),
}

# metricas en las que un valor mas alto es mejor (las demas son errores a minimizar)
METRICAS_MAYOR_ES_MEJOR = frozenset({"R2", "spearman"})


class EstratificadoPorCuantiles:
    """`StratifiedKFold` para regresión: estratifica por cuantiles del objetivo.

    `StratifiedKFold` necesita clases, y aquí el objetivo es continuo. Discretizarlo en
    cuantiles conserva su intención —que cada pliegue tenga la misma mezcla de aspirantes
    fuertes, medios y flojos— y evita el pliegue desafortunado que se lleva casi todos los
    valores altos, que con 353 filas de entrenamiento es un riesgo real.
    """

    def __init__(self, n_splits: int = 5, n_grupos: int = 4, semilla: int = 42) -> None:
        self.n_splits = n_splits
        self.n_grupos = n_grupos
        self.semilla = semilla

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        """Número de pliegues, como exige la interfaz de scikit-learn."""
        return self.n_splits

    def split(
        self, X: pd.DataFrame, y: pd.Series, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Genera los pliegues estratificando por el cuantil del objetivo."""
        grupos = pd.qcut(pd.Series(y).rank(method="first"), self.n_grupos, labels=False)
        interno = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.semilla)
        yield from interno.split(X, grupos)


VALIDADORES = ("kfold", "repetido", "estratificado", "series_temporales")


def construir_validador(
    nombre: str = "repetido",
    *,
    pliegues: int = 5,
    repeticiones: int = 5,
    semilla: int = 42,
) -> object:
    """Devuelve el esquema de validación cruzada pedido.

    - `kfold`: el más simple, cinco pliegues barajados.
    - `repetido` (por defecto): repite el `KFold` con particiones distintas. Con 353 filas
      de entrenamiento, un solo `KFold` da una media con mucha varianza; repetirlo cinco
      veces convierte 5 mediciones en 25 y estabiliza la estimación sin coste real.
    - `estratificado`: reparte por cuantiles del objetivo (ver `EstratificadoPorCuantiles`).
    - `series_temporales`: entrena siempre con el pasado y valida con el futuro. **No
      aplica a este dataset**, que no tiene columna temporal ni orden entre aspirantes; se
      ofrece porque el día que lleguen datos por convocatoria será el único correcto.
    """
    if nombre == "kfold":
        return KFold(n_splits=pliegues, shuffle=True, random_state=semilla)
    if nombre == "repetido":
        return RepeatedKFold(n_splits=pliegues, n_repeats=repeticiones, random_state=semilla)
    if nombre == "estratificado":
        return EstratificadoPorCuantiles(n_splits=pliegues, semilla=semilla)
    if nombre == "series_temporales":
        return TimeSeriesSplit(n_splits=pliegues)
    raise ValueError(f"Validador '{nombre}' desconocido. Disponibles: {list(VALIDADORES)}")


def _media(valores: np.ndarray) -> float:
    """Media ignorando nulos, o `nan` si no queda ningún valor.

    `np.nanmean` sobre una serie entera de nulos funciona, pero avisa por consola; aquí
    todos los pliegues con `spearman` nula son un resultado esperado (un modelo constante
    no ordena), no algo de lo que haya que avisar en cada ejecución.
    """
    finitos = valores[~np.isnan(valores)]
    return float(finitos.mean()) if finitos.size else float("nan")


def _desviacion(valores: np.ndarray) -> float:
    """Desviación típica ignorando nulos, o `nan` si no queda ningún valor."""
    finitos = valores[~np.isnan(valores)]
    return float(finitos.std()) if finitos.size else float("nan")


def _a_positivo(metrica: str, valor: float) -> float:
    """Convierte las puntuaciones `neg_*` de scikit-learn en el error que representan."""
    if metrica in METRICAS_MAYOR_ES_MEJOR:
        return float(valor)
    return float(-valor)


@dataclass(frozen=True)
class ConfiguracionValidacion:
    """Cómo se ejecutó la validación. Se guarda con los resultados para reproducirla."""

    validador: str = "repetido"
    pliegues: int = 5
    repeticiones: int = 5
    semilla: int = 42

    def sin_repeticiones(self) -> "ConfiguracionValidacion":
        """La misma configuración sin repetir la partición, para la curva de aprendizaje.

        La curva ya ajusta el modelo una vez por talla y pliegue; repetirlo cinco veces
        multiplicaría el coste sin cambiar su forma, que es lo que se mira.
        """
        if self.validador != "repetido":
            return self
        return ConfiguracionValidacion("kfold", self.pliegues, 1, self.semilla)

    def construir(self) -> object:
        """Instancia el esquema de validación cruzada descrito por esta configuración."""
        return construir_validador(
            self.validador,
            pliegues=self.pliegues,
            repeticiones=self.repeticiones,
            semilla=self.semilla,
        )


def validar_con_cruzada(
    modelo: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    configuracion: ConfiguracionValidacion | None = None,
) -> pd.DataFrame:
    """Validación cruzada sobre entrenamiento, con todas las métricas a la vez.

    Devuelve una fila por métrica con su valor en entrenamiento y en validación, la
    desviación entre pliegues y la brecha entre ambos. Medir train y validación en la misma
    pasada es lo que permite distinguir un modelo que no aprende de uno que memoriza: los
    dos dan mal resultado en validación, pero solo el segundo acierta en entrenamiento.
    """
    config = configuracion or ConfiguracionValidacion()
    resultado = cross_validate(
        clone(modelo),
        X_train,
        y_train,
        cv=config.construir(),
        scoring=METRICAS_REGRESION,
        return_train_score=True,
    )

    filas = []
    for metrica in METRICAS_REGRESION:
        validacion = np.array([_a_positivo(metrica, v) for v in resultado[f"test_{metrica}"]])
        entrenamiento = np.array([_a_positivo(metrica, v) for v in resultado[f"train_{metrica}"]])
        filas.append(
            {
                "metrica": metrica,
                "entrenamiento": _media(entrenamiento),
                "validacion_cruzada": _media(validacion),
                "desviacion_cv": _desviacion(validacion),
                "brecha_train_cv": _media(validacion) - _media(entrenamiento),
                "validador": config.validador,
                "pliegues": config.pliegues,
                "repeticiones": config.repeticiones if config.validador == "repetido" else 1,
                "semilla": config.semilla,
            }
        )
    return pd.DataFrame(filas)


def comparar_train_cv_test(
    validacion: pd.DataFrame, metricas_prueba: dict[str, float]
) -> pd.DataFrame:
    """Une los tres escenarios en una sola tabla: entrenamiento, validación y prueba.

    `diferencia_cv_prueba` es la columna que más dice: si la prueba se aparta mucho de la
    validación cruzada, el número de prueba es suerte —buena o mala— de esa partición
    concreta, y no debería comunicarse como si fuera el rendimiento esperado.
    """
    comparacion = validacion.copy()
    comparacion["prueba"] = comparacion["metrica"].map(metricas_prueba)
    comparacion["diferencia_cv_prueba"] = comparacion["prueba"] - comparacion["validacion_cruzada"]
    columnas = [
        "metrica",
        "entrenamiento",
        "validacion_cruzada",
        "desviacion_cv",
        "prueba",
        "brecha_train_cv",
        "diferencia_cv_prueba",
        "validador",
        "pliegues",
        "repeticiones",
        "semilla",
    ]
    return comparacion[[columna for columna in columnas if columna in comparacion.columns]]


def _valor(comparacion: pd.DataFrame, metrica: str, columna: str) -> float:
    """Lee una celda de la tabla de comparación, o `nan` si esa métrica no está."""
    fila = comparacion.loc[comparacion["metrica"] == metrica, columna]
    return float(fila.iloc[0]) if not fila.empty else float("nan")


def _diagnosticar_ajuste(mae_train: float, mae_cv: float) -> tuple[str, str, str]:
    """Veredicto de subajuste/sobreajuste a partir de la brecha relativa."""
    brecha_relativa = (mae_cv - mae_train) / mae_cv if mae_cv else 0.0
    evidencia = (
        f"MAE {mae_train:.4f} en entrenamiento y {mae_cv:.4f} en validacion "
        f"({brecha_relativa:.0%} del error de validacion es brecha)"
    )
    if brecha_relativa >= BRECHA_SOBREAJUSTE:
        return (
            "sobreajuste",
            evidencia,
            "limitar la capacidad del modelo (profundidad, min_samples_leaf), "
            "regularizar o conseguir mas datos",
        )
    if brecha_relativa >= BRECHA_SOBREAJUSTE_LEVE:
        return (
            "sobreajuste leve",
            evidencia,
            "vigilar: aceptable si la validacion cruzada supera claramente al modelo base",
        )
    return ("ajuste adecuado", evidencia, "ninguna: el modelo rinde igual dentro y fuera")


def diagnosticar_generalizacion(comparacion: pd.DataFrame) -> pd.DataFrame:
    """Convierte la tabla de métricas en veredictos con acciones recomendadas.

    Cuatro preguntas, cada una con su umbral documentado arriba: ¿memoriza?, ¿aprende algo?,
    ¿es estable entre pliegues? y ¿la prueba confirma lo que dijo la validación cruzada?
    """
    mae_train = _valor(comparacion, "MAE", "entrenamiento")
    mae_cv = _valor(comparacion, "MAE", "validacion_cruzada")
    mae_prueba = _valor(comparacion, "MAE", "prueba")
    desviacion = _valor(comparacion, "MAE", "desviacion_cv")
    r2_cv = _valor(comparacion, "R2", "validacion_cruzada")

    veredicto_ajuste, evidencia_ajuste, accion_ajuste = _diagnosticar_ajuste(mae_train, mae_cv)
    filas = [
        {
            "aspecto": "sobreajuste",
            "veredicto": veredicto_ajuste,
            "evidencia": evidencia_ajuste,
            "accion_recomendada": accion_ajuste,
        }
    ]

    subajuste = not np.isnan(r2_cv) and r2_cv < R2_SUBAJUSTE
    filas.append(
        {
            "aspecto": "subajuste",
            "veredicto": "subajuste" if subajuste else "capacidad suficiente",
            "evidencia": f"R2 de validacion cruzada {r2_cv:.4f} (umbral {R2_SUBAJUSTE})",
            "accion_recomendada": (
                "mas capacidad (modelo no lineal), mejores atributos o revisar el preprocesamiento"
                if subajuste
                else "ninguna: el modelo explica la mayor parte de la varianza"
            ),
        }
    )

    variacion = desviacion / mae_cv if mae_cv else 0.0
    inestable = variacion > INESTABILIDAD
    filas.append(
        {
            "aspecto": "estabilidad",
            "veredicto": "inestable" if inestable else "estable",
            "evidencia": (
                f"desviacion entre pliegues {desviacion:.4f}, un {variacion:.0%} del MAE medio"
            ),
            "accion_recomendada": (
                "aumentar repeticiones de la validacion cruzada y conseguir mas datos: "
                "con esta varianza, comparar modelos por decimas no significa nada"
                if inestable
                else "ninguna: los pliegues coinciden entre si"
            ),
        }
    )

    degradacion = abs(mae_prueba - mae_cv) / mae_cv if mae_cv and not np.isnan(mae_prueba) else 0.0
    degrada = degradacion > DEGRADACION_MAXIMA
    filas.append(
        {
            "aspecto": "degradacion_cv_prueba",
            "veredicto": "la prueba no confirma la validacion" if degrada else "coherente",
            "evidencia": (
                f"MAE {mae_cv:.4f} en validacion cruzada y {mae_prueba:.4f} en prueba "
                f"({degradacion:.0%} de diferencia relativa, umbral {DEGRADACION_MAXIMA:.0%})"
            ),
            "accion_recomendada": (
                "comunicar el resultado de la validacion cruzada, no el de prueba, y "
                "repetir la particion con otras semillas antes de decidir"
                if degrada
                else "ninguna: el conjunto de prueba se comporta como los pliegues"
            ),
        }
    )
    return pd.DataFrame(filas)


def curva_de_aprendizaje(
    modelo: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    configuracion: ConfiguracionValidacion | None = None,
    tallas: Sequence[float] = TALLAS_CURVA,
) -> pd.DataFrame:
    """Error en entrenamiento y validación según cuántas muestras se usan.

    Es la evidencia que responde a la pregunta cara: **¿esto se arregla con más datos?** Si
    las dos curvas ya se juntaron y se aplanaron, conseguir más filas no cambiará nada y hay
    que mirar a los atributos o al modelo. Si siguen separadas y bajando, sí.
    """
    config = configuracion or ConfiguracionValidacion()
    muestras, puntaje_train, puntaje_validacion = learning_curve(
        clone(modelo),
        X_train,
        y_train,
        train_sizes=list(tallas),
        cv=config.construir(),
        scoring="neg_mean_absolute_error",
        shuffle=True,
        random_state=config.semilla,
    )
    return pd.DataFrame(
        {
            "muestras_entrenamiento": muestras,
            "MAE_entrenamiento": -puntaje_train.mean(axis=1),
            "MAE_validacion": -puntaje_validacion.mean(axis=1),
            "desviacion_validacion": puntaje_validacion.std(axis=1),
        }
    )


def guardar_grafico_curva(curva: pd.DataFrame, ruta: Path) -> Path | None:
    """Dibuja la curva de aprendizaje si `matplotlib` está disponible.

    `matplotlib` es una dependencia de desarrollo, no de ejecución: el gráfico es un extra
    para quien analiza, y un pipeline que se cae porque falta una librería de dibujo sería
    un mal diseño. Si no está, se avisa y se sigue: la tabla de la curva ya está guardada.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "No se genero el grafico de la curva de aprendizaje: falta matplotlib "
            "(instalalo con 'uv sync --all-groups'). La tabla si esta guardada."
        )
        return None

    figura, ejes = plt.subplots(figsize=(7, 4))
    ejes.plot(
        curva["muestras_entrenamiento"], curva["MAE_entrenamiento"], "o-", label="entrenamiento"
    )
    ejes.plot(curva["muestras_entrenamiento"], curva["MAE_validacion"], "o-", label="validacion")
    ejes.fill_between(
        curva["muestras_entrenamiento"],
        curva["MAE_validacion"] - curva["desviacion_validacion"],
        curva["MAE_validacion"] + curva["desviacion_validacion"],
        alpha=0.15,
    )
    ejes.set_xlabel("muestras de entrenamiento")
    ejes.set_ylabel("MAE")
    ejes.set_title("Curva de aprendizaje")
    ejes.legend()
    figura.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=120)
    plt.close(figura)
    return ruta


def segmentos_debiles(
    modelo: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    columnas: Sequence[str] | None = None,
    n_grupos: int = 3,
) -> pd.DataFrame:
    """Error por segmento del conjunto de prueba: ¿a quién le funciona peor el modelo?

    El error medio esconde a quién perjudica. Se parte la prueba en tercios por cada
    atributo y se compara el MAE de cada tercio con el global; los que lo superan por más de
    `FACTOR_SEGMENTO_DEBIL` quedan marcados como débiles. Es el mismo análisis de segmentos
    que propone la guía con `deepchecks`, y el que en `07-interpretation` reveló que el
    modelo falla más con los aspirantes de probabilidad baja.
    """
    prediccion = modelo.predict(X_test)
    error_global = float(mean_absolute_error(y_test, prediccion))
    seleccionadas = list(columnas) if columnas is not None else list(X_test.columns)

    filas = []
    for columna in seleccionadas:
        valores = pd.to_numeric(X_test[columna], errors="coerce")
        if valores.nunique(dropna=True) < MIN_VALORES_PARA_SEGMENTAR:
            continue
        try:
            grupos = pd.qcut(valores.rank(method="first"), n_grupos, labels=False)
        except ValueError:  # pragma: no cover - solo con columnas degeneradas
            continue
        for grupo in sorted(pd.Series(grupos).dropna().unique()):
            mascara = (grupos == grupo).to_numpy()
            if mascara.sum() == 0:
                continue
            error = float(mean_absolute_error(y_test[mascara], prediccion[mascara]))
            filas.append(
                {
                    "atributo": columna,
                    "segmento": f"tercio {int(grupo) + 1}",
                    "n": int(mascara.sum()),
                    "MAE": error,
                    "MAE_global": error_global,
                    "razon_vs_global": error / error_global if error_global else float("nan"),
                    "debil": error > FACTOR_SEGMENTO_DEBIL * error_global,
                }
            )
    return pd.DataFrame(filas).sort_values("MAE", ascending=False).reset_index(drop=True)
