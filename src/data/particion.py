"""Chequeos de la partición train/test: fuga de información y representatividad.

Responde a dos preguntas que deciden si una evaluación significa algo:

1. **¿El modelo ya vio los datos con los que se le va a examinar?** Si una fila de prueba
   estaba en entrenamiento, la métrica mide memoria, no aprendizaje.
2. **¿Los dos conjuntos hablan del mismo problema?** Si la distribución de prueba no se
   parece a la de entrenamiento, el número que salga no dice nada sobre el rendimiento
   futuro —ni bueno ni malo—, simplemente no es comparable.

Los chequeos siguen la lista de la [guía de train/test checks][guia]: índices solapados,
muestras compartidas, proporción de la partición, deriva por característica con
Kolmogorov-Smirnov, categorías nuevas en prueba, deriva del objetivo y deriva multivariante
con un clasificador de dominio.

[guia]: https://joserzapata.github.io/courses/ciencia-datos-en-produccion/data-validation/train_test-checks/

**Por qué no `deepchecks` ni `evidently`.** Son las herramientas que sugiere la guía y
harían exactamente esto; el criterio, como en `src/data/validacion.py`, fue no añadir una
dependencia pesada para siete comprobaciones que caben en un módulo con pruebas, cuando el
proyecto ya trae `scipy` y `scikit-learn`. Los umbrales son los mismos que usa la suite de
`deepchecks`, así que migrar no cambiaría ningún veredicto.

**Dos severidades, y la diferencia importa.** La fuga de información es un `error`: invalida
la evaluación y detiene el pipeline. La deriva es una `advertencia`: con 471 filas, una
partición aleatoria produce diferencias por azar, y bloquear el entrenamiento por eso sería
ruido. Se registra, se guarda en el informe y quien quiera rigor máximo usa `estricto=True`.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data.validacion import ErrorValidacion, Violacion

logger = logging.getLogger(__name__)

OK = "ok"
ADVERTENCIA = "advertencia"
ERROR = "error"

# proporcion de filas de prueba que pueden aparecer en entrenamiento antes de considerarlo
# fuga grave: el mismo 5 % que usa la suite de deepchecks
MAX_SOLAPE = 0.05

# estadistico D de Kolmogorov-Smirnov por encima del cual se considera que una variable
# cambio de distribucion entre los dos conjuntos
UMBRAL_KS = 0.2

# AUC de un clasificador que intenta distinguir train de test. 0.5 es no distinguirlos en
# absoluto —lo deseable—; por encima de 0.65 los conjuntos son reconocibles
UMBRAL_AUC_DOMINIO = 0.65

# desviacion admitida entre la proporcion de prueba pedida y la obtenida
TOLERANCIA_PROPORCION = 0.05

# ratio minimo test/train (deepchecks) y tamano minimo para que una metrica signifique algo
RATIO_MINIMO = 0.01
MIN_FILAS_TEST = 30

# diferencia admitida entre la proporcion de nulos de cada conjunto, en puntos
MAX_DIFERENCIA_NULOS = 0.10


@dataclass(frozen=True)
class Particion:
    """Los cuatro conjuntos que produce una separación train/test.

    Agruparlos en un objeto evita arrastrar cuatro parámetros por todas las firmas y deja
    claro que son **una sola cosa**: una partición concreta, con su semilla y su proporción.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        """Permite seguir desempaquetándola como la tupla de `train_test_split`."""
        return iter((self.X_train, self.X_test, self.y_train, self.y_test))


@dataclass(frozen=True)
class ResultadoChequeo:
    """El veredicto de un chequeo, con el número que lo sustenta.

    Se guarda también cuando pasa (`severidad = "ok"`): un informe que solo enumera
    problemas no deja constancia de lo que sí se comprobó.
    """

    chequeo: str
    columna: str
    severidad: str
    detalle: str
    estadistico: float | None = None
    umbral: float | None = None

    def __str__(self) -> str:
        ubicacion = f"[{self.columna}] " if self.columna else ""
        medida = f" ({self.estadistico:.4f} vs. umbral {self.umbral})" if self.umbral else ""
        return f"{ubicacion}{self.chequeo}: {self.detalle}{medida}"


def _columnas_numericas(datos: pd.DataFrame) -> list[str]:
    """Columnas sobre las que tiene sentido aplicar Kolmogorov-Smirnov."""
    return [columna for columna in datos.columns if pd.api.types.is_numeric_dtype(datos[columna])]


def chequear_indices(X_train: pd.DataFrame, X_test: pd.DataFrame) -> ResultadoChequeo:
    """Ningún índice puede estar en los dos conjuntos: es fuga de información directa."""
    comunes = X_train.index.intersection(X_test.index)
    if len(comunes) == 0:
        return ResultadoChequeo(
            "indices_solapados", "", OK, "ninguna fila esta en los dos conjuntos"
        )
    return ResultadoChequeo(
        "indices_solapados",
        "",
        ERROR,
        f"{len(comunes)} indices aparecen en train y en test: {comunes[:5].tolist()}",
        float(len(comunes)),
        0.0,
    )


def chequear_filas_compartidas(X_train: pd.DataFrame, X_test: pd.DataFrame) -> ResultadoChequeo:
    """Perfiles idénticos en ambos conjuntos, aunque tengan índices distintos.

    Sin identificador de aspirante, dos filas iguales son el mismo caso: el modelo lo
    memorizó en entrenamiento y acertará en prueba sin haber aprendido nada.
    """
    if len(X_test) == 0:
        return ResultadoChequeo("filas_compartidas", "", OK, "el conjunto de prueba esta vacio")
    claves_train = set(map(tuple, X_train.to_numpy().tolist()))
    compartidas = sum(1 for fila in X_test.to_numpy().tolist() if tuple(fila) in claves_train)
    proporcion = compartidas / len(X_test)
    if compartidas == 0:
        return ResultadoChequeo(
            "filas_compartidas", "", OK, "ningun perfil de prueba aparece en entrenamiento", 0.0
        )
    severidad = ERROR if proporcion > MAX_SOLAPE else ADVERTENCIA
    return ResultadoChequeo(
        "filas_compartidas",
        "",
        severidad,
        f"{compartidas} de {len(X_test)} perfiles de prueba ({proporcion:.1%}) ya estaban en train",
        proporcion,
        MAX_SOLAPE,
    )


def chequear_tamanos(
    X_train: pd.DataFrame, X_test: pd.DataFrame, proporcion_esperada: float
) -> list[ResultadoChequeo]:
    """La partición debe tener el tamaño pedido y dejar prueba suficiente para medir."""
    total = len(X_train) + len(X_test)
    proporcion = len(X_test) / total if total else 0.0
    resultados = []

    desviacion = abs(proporcion - proporcion_esperada)
    resultados.append(
        ResultadoChequeo(
            "proporcion_de_la_particion",
            "",
            OK if desviacion <= TOLERANCIA_PROPORCION else ADVERTENCIA,
            f"prueba = {proporcion:.1%} del total, se pidio {proporcion_esperada:.1%}",
            desviacion,
            TOLERANCIA_PROPORCION,
        )
    )

    ratio = len(X_test) / len(X_train) if len(X_train) else 0.0
    if ratio < RATIO_MINIMO or len(X_test) < MIN_FILAS_TEST:
        resultados.append(
            ResultadoChequeo(
                "tamano_del_conjunto_de_prueba",
                "",
                ADVERTENCIA,
                f"{len(X_test)} filas de prueba (ratio {ratio:.3f}): una metrica sobre tan "
                f"pocos datos tiene un intervalo enorme",
                float(len(X_test)),
                float(MIN_FILAS_TEST),
            )
        )
    else:
        resultados.append(
            ResultadoChequeo(
                "tamano_del_conjunto_de_prueba",
                "",
                OK,
                f"{len(X_test)} filas de prueba frente a {len(X_train)} de entrenamiento",
                float(len(X_test)),
                float(MIN_FILAS_TEST),
            )
        )
    return resultados


def chequear_deriva_por_columna(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> list[ResultadoChequeo]:
    """Kolmogorov-Smirnov por variable: ¿cambia su distribución entre los dos conjuntos?

    El estadístico D es la máxima distancia entre las dos distribuciones acumuladas, así que
    no depende del tamaño de la muestra como sí lo hace el p-valor. Se compara con 0.2, el
    umbral de la suite de `deepchecks`.
    """
    resultados = []
    for columna in _columnas_numericas(X_train):
        if columna not in X_test.columns:
            continue
        valores_train = X_train[columna].dropna().to_numpy(dtype=float)
        valores_test = X_test[columna].dropna().to_numpy(dtype=float)
        if len(valores_train) == 0 or len(valores_test) == 0:
            continue
        estadistico = float(stats.ks_2samp(valores_train, valores_test).statistic)
        resultados.append(
            ResultadoChequeo(
                "deriva_de_la_variable",
                columna,
                OK if estadistico <= UMBRAL_KS else ADVERTENCIA,
                "distancia KS entre las distribuciones de train y test",
                estadistico,
                UMBRAL_KS,
            )
        )
    return resultados


def chequear_nulos(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[ResultadoChequeo]:
    """Los datos ausentes deben repartirse parecido en los dos conjuntos.

    Una columna con 2 % de nulos en train y 20 % en test significa que el modelo se evalúa
    sobre perfiles mucho más incompletos de los que aprendió a tratar.
    """
    resultados = []
    for columna in X_train.columns:
        if columna not in X_test.columns:
            continue
        diferencia = abs(float(X_train[columna].isna().mean() - X_test[columna].isna().mean()))
        if diferencia > MAX_DIFERENCIA_NULOS:
            resultados.append(
                ResultadoChequeo(
                    "deriva_de_los_nulos",
                    columna,
                    ADVERTENCIA,
                    f"la proporcion de nulos difiere {diferencia:.1%} entre train y test",
                    diferencia,
                    MAX_DIFERENCIA_NULOS,
                )
            )
    return resultados


def chequear_categorias_nuevas(
    X_train: pd.DataFrame, X_test: pd.DataFrame, columnas_categoricas: list[str]
) -> list[ResultadoChequeo]:
    """Una categoría que solo aparece en prueba es una que el modelo nunca aprendió."""
    resultados = []
    for columna in columnas_categoricas:
        if columna not in X_train.columns or columna not in X_test.columns:
            continue
        nuevas = sorted(set(X_test[columna].dropna()) - set(X_train[columna].dropna()))
        resultados.append(
            ResultadoChequeo(
                "categorias_nuevas_en_prueba",
                columna,
                OK if not nuevas else ADVERTENCIA,
                "sin categorias nuevas" if not nuevas else f"solo en prueba: {nuevas[:5]}",
                float(len(nuevas)),
                0.0,
            )
        )
    return resultados


def chequear_deriva_del_objetivo(y_train: pd.Series, y_test: pd.Series) -> ResultadoChequeo:
    """El objetivo debe distribuirse igual en los dos conjuntos.

    Es el chequeo más importante de los de deriva: si el objetivo de prueba está desplazado,
    hasta un modelo perfecto parecerá sesgado.
    """
    estadistico = float(
        stats.ks_2samp(y_train.to_numpy(dtype=float), y_test.to_numpy(dtype=float)).statistic
    )
    diferencia_medias = abs(float(y_train.mean() - y_test.mean()))
    return ResultadoChequeo(
        "deriva_del_objetivo",
        str(y_train.name or "objetivo"),
        OK if estadistico <= UMBRAL_KS else ADVERTENCIA,
        f"medias {y_train.mean():.3f} y {y_test.mean():.3f} (diferencia {diferencia_medias:.3f})",
        estadistico,
        UMBRAL_KS,
    )


def chequear_deriva_multivariante(
    X_train: pd.DataFrame, X_test: pd.DataFrame, semilla: int = 42
) -> ResultadoChequeo:
    """Clasificador de dominio: ¿se puede distinguir una fila de train de una de test?

    Se entrena una regresión logística para predecir a qué conjunto pertenece cada fila. Si
    no lo consigue (AUC ≈ 0.5), la partición es aleatoria de verdad. Un AUC alto delata una
    diferencia **conjunta** que los chequeos por columna no ven: cada variable por separado
    puede parecer igual y aun así las combinaciones ser distintas.
    """
    columnas = [col for col in _columnas_numericas(X_train) if col in X_test.columns]
    if not columnas or min(len(X_train), len(X_test)) < MIN_FILAS_TEST:
        return ResultadoChequeo(
            "deriva_multivariante", "", OK, "datos insuficientes para el clasificador de dominio"
        )

    X = pd.concat([X_train[columnas], X_test[columnas]], ignore_index=True)
    X = X.fillna(X.median(numeric_only=True))
    etiqueta = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    modelo = LogisticRegression(max_iter=1000, random_state=semilla)
    auc = float(
        cross_val_score(
            modelo,
            X,
            etiqueta,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=semilla),
            scoring="roc_auc",
        ).mean()
    )
    return ResultadoChequeo(
        "deriva_multivariante",
        "",
        OK if auc <= UMBRAL_AUC_DOMINIO else ADVERTENCIA,
        "AUC de un clasificador que intenta distinguir train de test (0.5 = indistinguibles)",
        auc,
        UMBRAL_AUC_DOMINIO,
    )


def revisar_particion(
    particion: Particion,
    *,
    proporcion_esperada: float = 0.25,
    columnas_categoricas: list[str] | None = None,
    semilla: int = 42,
) -> list[ResultadoChequeo]:
    """Ejecuta todos los chequeos y devuelve sus resultados, incluidos los que pasan.

    No lanza excepciones: sirve para inspeccionar la partición desde un notebook o para
    construir el informe. Quien quiera que el proceso se detenga usa
    `validar_particion_train_test`.
    """
    X_train, X_test = particion.X_train, particion.X_test
    y_train, y_test = particion.y_train, particion.y_test
    categoricas = (
        columnas_categoricas
        if columnas_categoricas is not None
        else [col for col in ("university_rating", "research") if col in X_train.columns]
    )
    resultados = [
        chequear_indices(X_train, X_test),
        chequear_filas_compartidas(X_train, X_test),
        *chequear_tamanos(X_train, X_test, proporcion_esperada),
        chequear_deriva_del_objetivo(y_train, y_test),
        *chequear_deriva_por_columna(X_train, X_test),
        *chequear_categorias_nuevas(X_train, X_test, categoricas),
        *chequear_nulos(X_train, X_test),
        chequear_deriva_multivariante(X_train, X_test, semilla),
    ]
    return resultados


def informe_particion(resultados: list[ResultadoChequeo]) -> pd.DataFrame:
    """Tabla con el resultado de cada chequeo, lista para guardar junto a las métricas."""
    return pd.DataFrame(
        [
            {
                "chequeo": resultado.chequeo,
                "columna": resultado.columna,
                "severidad": resultado.severidad,
                "estadistico": resultado.estadistico,
                "umbral": resultado.umbral,
                "detalle": resultado.detalle,
            }
            for resultado in resultados
        ]
    )


def validar_particion_train_test(
    particion: Particion,
    *,
    proporcion_esperada: float = 0.25,
    estricto: bool = False,
    semilla: int = 42,
) -> pd.DataFrame:
    """Valida la partición y devuelve el informe completo de los chequeos.

    Es la función independiente que pide el issue (`validate_train_test_split`), en la
    convención de nombres del repositorio. Se puede usar suelta, sin el pipeline:

    ```python
    informe = validar_particion_train_test(Particion(X_train, X_test, y_train, y_test))
    ```

    Parameters
    ----------
    estricto:
        si es `True`, cualquier advertencia detiene el proceso. Por defecto solo lo detiene
        la fuga de información, que es la que invalida la evaluación.

    Raises
    ------
    ErrorValidacion
        si algún chequeo es de severidad `error` —o `advertencia` con `estricto=True`—, con
        el detalle de cada problema.
    """
    resultados = revisar_particion(
        particion,
        proporcion_esperada=proporcion_esperada,
        semilla=semilla,
    )
    informe = informe_particion(resultados)

    for resultado in resultados:
        if resultado.severidad == ADVERTENCIA:
            logger.warning("Chequeo de la particion: %s", resultado)

    graves = [
        resultado
        for resultado in resultados
        if resultado.severidad == ERROR or (estricto and resultado.severidad == ADVERTENCIA)
    ]
    if graves:
        raise ErrorValidacion(
            [
                Violacion(resultado.chequeo, resultado.columna, resultado.detalle)
                for resultado in graves
            ],
            contexto="la particion train/test",
        )
    return informe
