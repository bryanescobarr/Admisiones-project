"""Training pipeline del proyecto de admisiones: de las features al modelo entrenado.

Segunda etapa de la arquitectura FTI de `src/README.md`. Lee las features que dejó el
feature pipeline, separa entrenamiento y prueba, entrena, evalúa y guarda el modelo junto
con sus métricas.

    data/04_feature/admisiones_features.parquet
        -> data/06_models/modelo_training_pipeline.joblib
        -> data/08_reporting/metricas_training_pipeline.parquet

**La partición se comprueba antes de entrenar.** Entre partir y ajustar hay una puerta:
`chequear_particion` verifica que ningún perfil de prueba estuvo en entrenamiento y que los
dos conjuntos representan la misma distribución (`src/data/particion.py`). Una fuga detiene
el pipeline sin dejar artefactos; la deriva se avisa y se registra en el informe.

**El orden protege contra la fuga de datos.** La partición ocurre antes que cualquier
ajuste, y la imputación y el escalado viven dentro del `Pipeline`, así que sus parámetros
—la mediana, la media, la desviación— se aprenden **solo con el conjunto de
entrenamiento**. Es la razón por la que el feature pipeline deja esos pasos sin hacer: el
único sitio donde pueden ajustarse sin contaminar la evaluación es aquí, después de partir.

**La validación no se queda en una cifra de prueba.** Cada ejecución mide las mismas
métricas en entrenamiento, en validación cruzada sobre *train* y en prueba, y traduce esa
comparación en un diagnóstico explícito de subajuste, sobreajuste, estabilidad y
degradación, con la acción recomendada para cada uno (`src/model/validacion.py`). La
evidencia —tablas, curva de aprendizaje y segmentos débiles— queda guardada en
`data/08_reporting/`.

**El modelo entrenado no se evalúa contra sí mismo.** Cada ejecución mide también las dos
referencias de `5-models` —la media del objetivo y la heurística manual de `03.3`— sobre la
misma partición. Un MAE de 0.047 no significa nada solo; comparado con el 0.119 del modelo
trivial y el 0.064 de la regla manual, sí.

El artefacto se guarda aparte del modelo que sirve la demo
(`data/06_models/modelo_final_automl.joblib`): una ejecución de este script no debe cambiar
lo que ven los usuarios sin que alguien lo decida.

Uso:

    uv run python src/pipelines/training_pipeline/train_pipeline.py
    uv run python src/pipelines/training_pipeline/train_pipeline.py --modelo ridge
    uv run python src/pipelines/training_pipeline/train_pipeline.py --modelo automl --presupuesto 120
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# el script se ejecuta directamente, asi que `src/` no esta en sys.path todavia
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.particion import Particion, validar_particion_train_test  # noqa: E402
from data.preprocesamiento import construir_preprocesamiento  # noqa: E402
from data.validacion import ErrorValidacion  # noqa: E402
from model.heuristica import HeuristicaAdmision  # noqa: E402
from model.validacion import (  # noqa: E402
    VALIDADORES,
    ConfiguracionValidacion,
    comparar_train_cv_test,
    curva_de_aprendizaje,
    diagnosticar_generalizacion,
    guardar_grafico_curva,
    segmentos_debiles,
    validar_con_cruzada,
)
from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    TARGET,
    buscar_raiz_proyecto,
    validar_features,
)

logger = logging.getLogger(__name__)

# mismos valores que en 04-feat_eng y 06-models, para que las metricas sean comparables
PROPORCION_TEST = 0.25
SEMILLA = 42
PLIEGUES_VALIDACION = 5
REPETICIONES_VALIDACION = 5

# presupuesto por defecto de la busqueda automatica, en segundos (solo con --modelo automl)
PRESUPUESTO_AUTOML = 180
FAMILIAS_AUTOML = ["lgbm", "xgboost", "rf", "extra_tree", "xgb_limitdepth"]

RUTA_FEATURES_RELATIVA = Path("data") / "04_feature" / "admisiones_features.parquet"
RUTA_MODELO_RELATIVA = Path("data") / "06_models" / "modelo_training_pipeline.joblib"
RUTA_METRICAS_RELATIVA = Path("data") / "08_reporting" / "metricas_training_pipeline.parquet"
RUTA_CHEQUEOS_RELATIVA = Path("data") / "08_reporting" / "chequeos_particion.parquet"
DIR_REPORTES_RELATIVA = Path("data") / "08_reporting"

# nombres de los artefactos de la validacion del modelo dentro del directorio de reportes
ARCHIVO_VALIDACION = "validacion_modelo.parquet"
ARCHIVO_DIAGNOSTICO = "diagnostico_generalizacion.parquet"
ARCHIVO_CURVA = "curva_aprendizaje.parquet"
ARCHIVO_GRAFICO_CURVA = "curva_aprendizaje.png"
ARCHIVO_SEGMENTOS = "segmentos_debiles.parquet"

# referencias que se miden en cada ejecucion junto al modelo elegido
MODELO_DE_REFERENCIA = "dummy"


def _extra_trees(semilla: int) -> BaseEstimator:
    """Familia elegida por la búsqueda automática de `06-models`, con sus valores por defecto.

    En aquel cuaderno la configuración ajustada por FLAML obtuvo 0.0476 de MAE en prueba y
    la misma familia sin ajustar, 0.0467: la diferencia es ruido en 471 filas. Se entrena la
    versión por defecto porque es **reproducible sin volver a buscar**, y quien quiera la
    búsqueda completa tiene `--modelo automl`.
    """
    return ExtraTreesRegressor(random_state=semilla, n_jobs=-1)


def _automl(semilla: int, presupuesto: int = PRESUPUESTO_AUTOML) -> BaseEstimator:
    """Búsqueda automática con FLAML, la misma configuración que `06-models`."""
    # import local a proposito: FLAML tarda segundos en cargarse y solo hace falta con
    # `--modelo automl`; arriba penalizaria cada ejecucion del script
    from flaml import AutoML  # noqa: PLC0415

    return AutoML(
        task="regression",
        metric="mae",
        eval_method="cv",
        n_splits=PLIEGUES_VALIDACION,
        time_budget=presupuesto,
        estimator_list=FAMILIAS_AUTOML,
        seed=semilla,
        verbose=0,
    )


# catalogo de modelos entrenables; `preprocesar=False` para el que consume datos crudos
CATALOGO_MODELOS: dict[str, dict[str, Any]] = {
    "dummy": {
        "descripcion": "media del objetivo (referencia trivial de 05-models)",
        "crear": lambda semilla: DummyRegressor(strategy="mean"),
        "preprocesar": True,
    },
    "heuristica": {
        "descripcion": "regla manual recalibrada de 03.3",
        "crear": lambda semilla: HeuristicaAdmision(recalibrar=True),
        "preprocesar": False,
    },
    "ridge": {
        "descripcion": "regresion lineal regularizada (referencia de 04-feat_eng)",
        "crear": lambda semilla: Ridge(alpha=1.0),
        "preprocesar": True,
    },
    "extra_trees": {
        "descripcion": "familia elegida por el AutoML de 06-models",
        "crear": _extra_trees,
        "preprocesar": True,
    },
    "automl": {
        "descripcion": "busqueda automatica con FLAML",
        "crear": _automl,
        "preprocesar": True,
    },
}


@dataclass(frozen=True)
class RutasEntrenamiento:
    """Las rutas que definen la entrada y las salidas de esta etapa."""

    features: Path
    modelo: Path
    metricas: Path
    chequeos: Path | None = None
    reportes: Path | None = None


@dataclass(frozen=True)
class ResultadoEntrenamiento:
    """Lo que produce una ejecución completa del training pipeline."""

    modelo: Pipeline
    metricas: pd.DataFrame
    ruta_modelo: Path
    ruta_metricas: Path
    chequeos_particion: pd.DataFrame | None = None
    validacion: pd.DataFrame | None = None
    diagnostico: pd.DataFrame | None = None
    curva: pd.DataFrame | None = None
    segmentos: pd.DataFrame | None = None


def leer_features(ruta: Path) -> pd.DataFrame:
    """Lee el parquet del feature pipeline y comprueba que cumple su contrato.

    La validación no se repite por desconfianza, sino porque **el archivo puede tener
    cualquier antigüedad**: nada garantiza que lo generara la versión actual del feature
    pipeline. El contrato de salida de una etapa es el contrato de entrada de la siguiente.
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontraron features en {ruta}. Ejecuta antes el feature pipeline: "
            "uv run python src/pipelines/feature_pipeline/feature_pipeline.py"
        )
    features = pd.read_parquet(ruta)
    logger.info("Leidas %d filas x %d columnas de %s", len(features), features.shape[1], ruta)
    return validar_features(features)


def separar_train_test(
    features: pd.DataFrame,
    proporcion_test: float = PROPORCION_TEST,
    semilla: int = SEMILLA,
) -> Particion:
    """Parte en entrenamiento y prueba **antes** de ajustar nada.

    Con 471 filas y un objetivo continuo sin cola pesada no hace falta estratificar: en
    `04-feat_eng` se comprobó que la media del objetivo queda casi idéntica en ambas partes
    (0.723 frente a 0.726).
    """
    X = features.drop(columns=[TARGET])
    y = features[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=proporcion_test, random_state=semilla
    )
    logger.info(
        "train: %d filas   test: %d filas   media del objetivo: %.3f / %.3f",
        len(X_train),
        len(X_test),
        y_train.mean(),
        y_test.mean(),
    )
    return Particion(X_train, X_test, y_train, y_test)


def construir_modelo(nombre: str, semilla: int = SEMILLA, **parametros: Any) -> Pipeline:
    """Arma el pipeline `preparacion + modelo` del catálogo.

    La heurística de `03.3` es la excepción: normaliza por el dominio documentado, así que
    **consume los datos crudos** y no debe recibir la salida del preprocesamiento. Se
    envuelve igualmente en un `Pipeline` para que el resto del script no tenga que
    distinguir un caso del otro.
    """
    if nombre not in CATALOGO_MODELOS:
        raise ValueError(f"Modelo '{nombre}' desconocido. Disponibles: {sorted(CATALOGO_MODELOS)}")
    definicion = CATALOGO_MODELOS[nombre]
    estimador = definicion["crear"](semilla, **parametros)
    pasos = []
    if definicion["preprocesar"]:
        pasos.append(("preparacion", construir_preprocesamiento()))
    pasos.append(("modelo", estimador))
    return Pipeline(pasos)


def entrenar(modelo: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Ajusta el pipeline completo sobre el conjunto de entrenamiento."""
    modelo.fit(X_train, y_train)
    logger.info("Modelo entrenado: %s", modelo.named_steps["modelo"].__class__.__name__)
    return modelo


def evaluar(modelo: Pipeline, X: pd.DataFrame, y: pd.Series, nombre: str) -> dict[str, Any]:
    """Métricas del modelo sobre un conjunto, con la interpretación de cada una.

    - **MAE**: métrica principal. Está en las unidades del objetivo (puntos de
      probabilidad), así que se le puede decir al usuario "nos equivocamos 5 puntos".
    - **RMSE**: penaliza los errores grandes, los que hacen descartar una universidad viable.
    - **R²** y **MAPE**: para comunicar, no para decidir.
    - **Spearman**: la que de verdad importa para el producto. El usuario ordena
      universidades; dos modelos con el mismo MAE pueden ordenar distinto.
    """
    prediccion = modelo.predict(X)
    # un modelo que predice siempre lo mismo -como el dummy- no ordena nada: la correlacion
    # de rangos no esta definida y se reporta como nula en vez de calcularla y avisar
    constante = float(np.max(prediccion) - np.min(prediccion)) == 0.0
    correlacion = float("nan") if constante else stats.spearmanr(y, prediccion).statistic
    return {
        "modelo": nombre,
        "MAE": float(mean_absolute_error(y, prediccion)),
        "RMSE": float(root_mean_squared_error(y, prediccion)),
        "R2": float(r2_score(y, prediccion)),
        "MAPE": float(mean_absolute_percentage_error(y, prediccion)),
        "spearman": float(correlacion),
    }


def validar_modelo(
    modelo: Pipeline,
    particion: Particion,
    metricas_prueba: dict[str, float],
    configuracion: ConfiguracionValidacion | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida la generalización del modelo y devuelve la comparación y el diagnóstico.

    Tres escenarios en una sola tabla —entrenamiento, validación cruzada y prueba— y, a
    partir de ella, un veredicto por aspecto con la acción recomendada. La métrica de
    prueba llega ya calculada porque el modelo entrenado es el mismo: repetirla aquí sería
    volver a predecir sobre las mismas filas.

    La validación cruzada se ejecuta sobre un clon **sin entrenar**, así que cada pliegue
    ajusta también el preprocesamiento con sus propios datos: usar el modelo ya ajustado
    filtraría el conjunto de validación de cada pliegue.
    """
    config = configuracion or ConfiguracionValidacion()
    validacion = validar_con_cruzada(modelo, particion.X_train, particion.y_train, config)
    comparacion = comparar_train_cv_test(validacion, metricas_prueba)
    diagnostico = diagnosticar_generalizacion(comparacion)

    logger.info(
        "Validacion cruzada (%s, %d pliegues x %d repeticiones, semilla %d):\n%s",
        config.validador,
        config.pliegues,
        config.repeticiones if config.validador == "repetido" else 1,
        config.semilla,
        comparacion.drop(columns=["validador", "pliegues", "repeticiones", "semilla"])
        .round(4)
        .to_string(index=False),
    )
    for _, fila in diagnostico.iterrows():
        logger.info(
            "Diagnostico [%s]: %s | %s", fila["aspecto"], fila["veredicto"], fila["evidencia"]
        )
    return comparacion, diagnostico


def resumen_de_validacion(comparacion: pd.DataFrame) -> dict[str, float]:
    """Extrae de la comparación las cifras que acompañan al modelo en la tabla de métricas."""
    mae = comparacion.loc[comparacion["metrica"] == "MAE"]
    if mae.empty:
        return {}
    fila = mae.iloc[0]
    return {
        "MAE_cv": float(fila["validacion_cruzada"]),
        "MAE_cv_desv": float(fila["desviacion_cv"]),
        "MAE_entrenamiento": float(fila["entrenamiento"]),
        "brecha_train_cv": float(fila["brecha_train_cv"]),
    }


def chequear_particion(
    particion: Particion,
    *,
    proporcion_test: float = PROPORCION_TEST,
    estricto: bool = False,
    semilla: int = SEMILLA,
) -> pd.DataFrame:
    """Comprueba la partición antes de entrenar y registra el resultado de cada chequeo.

    Entrenar sobre una partición con fuga de información es peor que no entrenar: produce
    métricas excelentes en las que alguien va a confiar. Por eso el chequeo ocurre **antes**
    del `fit` y una fuga detiene el pipeline sin dejar artefactos.

    La deriva, en cambio, se avisa y se registra: con 471 filas una partición aleatoria
    produce diferencias por azar, y bloquear el entrenamiento por eso sería ruido. Con
    `estricto=True`, cualquier advertencia detiene también el proceso.
    """
    informe = validar_particion_train_test(
        particion,
        proporcion_esperada=proporcion_test,
        estricto=estricto,
        semilla=semilla,
    )
    conteo = informe["severidad"].value_counts().to_dict()
    logger.info(
        "Chequeos de la particion: %d ejecutados, %d ok, %d advertencias",
        len(informe),
        conteo.get("ok", 0),
        conteo.get("advertencia", 0),
    )
    return informe


def construir_informe(evaluaciones: list[dict[str, Any]]) -> pd.DataFrame:
    """Tabla de métricas ordenada por MAE, con la mejora frente a la referencia trivial."""
    informe = pd.DataFrame(evaluaciones)
    referencia = informe.loc[informe["modelo"] == MODELO_DE_REFERENCIA, "MAE"]
    if not referencia.empty:
        mae_referencia = float(referencia.iloc[0])
        informe["mejora_vs_dummy_%"] = 100 * (mae_referencia - informe["MAE"]) / mae_referencia
    return informe.sort_values("MAE").reset_index(drop=True)


def guardar_modelo(modelo: Pipeline, ruta: Path, X_prueba: pd.DataFrame) -> Path:
    """Serializa el modelo y comprueba que al recargarlo predice exactamente lo mismo.

    Un artefacto que no se puede volver a cargar no es un modelo entrenado, es un archivo.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)
    recargado = joblib.load(ruta)
    if not np.allclose(recargado.predict(X_prueba), modelo.predict(X_prueba)):
        raise ValueError(f"El modelo guardado en {ruta} no reproduce las mismas predicciones")
    logger.info("Modelo guardado en %s (%.1f KB)", ruta, ruta.stat().st_size / 1024)
    return ruta


def _guardar_tabla(tabla: pd.DataFrame, ruta: Path, etiqueta: str) -> Path:
    """Escribe una tabla de resultados en parquet, junto al resto de reportes."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(ruta, index=False, engine="pyarrow", compression="snappy")
    logger.info("%s -> %s", etiqueta, ruta)
    return ruta


def guardar_metricas(metricas: pd.DataFrame, ruta: Path) -> Path:
    """Escribe la tabla de métricas en parquet, junto al resto de reportes del proyecto."""
    return _guardar_tabla(metricas, ruta, "Metricas")


def guardar_chequeos(chequeos: pd.DataFrame, ruta: Path) -> Path:
    """Escribe el informe de los chequeos de la partición.

    Se guarda siempre, también cuando todo pasa: el issue pide **los resultados** de las
    validaciones, y un informe que solo aparece cuando algo falla no deja constancia de lo
    que sí se comprobó en la ejecución que produjo el modelo.
    """
    return _guardar_tabla(chequeos, ruta, "Chequeos de la particion")


def _guardar_reportes_de_validacion(
    directorio: Path | None,
    comparacion: pd.DataFrame,
    diagnostico: pd.DataFrame,
    curva: pd.DataFrame | None,
    segmentos: pd.DataFrame,
) -> None:
    """Persiste la evidencia de la validación: comparación, diagnóstico, curva y segmentos.

    Se guarda siempre que haya directorio de reportes, pase o no el diagnóstico: la
    ejecución que produjo el modelo tiene que dejar por escrito cómo se validó.
    """
    if directorio is None:
        return
    _guardar_tabla(comparacion, directorio / ARCHIVO_VALIDACION, "Validacion del modelo")
    _guardar_tabla(diagnostico, directorio / ARCHIVO_DIAGNOSTICO, "Diagnostico de generalizacion")
    if curva is not None:
        _guardar_tabla(curva, directorio / ARCHIVO_CURVA, "Curva de aprendizaje")
    if not segmentos.empty:
        _guardar_tabla(
            segmentos, directorio / ARCHIVO_SEGMENTOS, "Segmentos del conjunto de prueba"
        )


def ejecutar_pipeline(  # noqa: PLR0913
    # cada parametro es una opcion de la linea de comandos: agruparlos en un objeto solo
    # moveria la lista de sitio y alejaria la firma de la interfaz que ve quien lo ejecuta
    rutas: RutasEntrenamiento,
    *,
    nombre_modelo: str = "extra_trees",
    semilla: int = SEMILLA,
    con_referencias: bool = True,
    proporcion_test: float = PROPORCION_TEST,
    particion_estricta: bool = False,
    configuracion_validacion: ConfiguracionValidacion | None = None,
    con_curva: bool = True,
    con_grafico: bool = False,
    **parametros_modelo: Any,
) -> ResultadoEntrenamiento:
    """Ejecuta el entrenamiento completo y devuelve el modelo con sus métricas."""
    features = leer_features(rutas.features)
    particion = separar_train_test(features, proporcion_test, semilla)
    X_train, X_test = particion.X_train, particion.X_test
    y_train, y_test = particion.y_train, particion.y_test
    chequeos = chequear_particion(
        particion,
        proporcion_test=proporcion_test,
        estricto=particion_estricta,
        semilla=semilla,
    )
    if rutas.chequeos is not None:
        guardar_chequeos(chequeos, rutas.chequeos)

    modelo = entrenar(
        construir_modelo(nombre_modelo, semilla, **parametros_modelo), X_train, y_train
    )
    evaluacion = evaluar(modelo, X_test, y_test, nombre_modelo)
    config_validacion = configuracion_validacion or ConfiguracionValidacion(semilla=semilla)
    comparacion, diagnostico = validar_modelo(
        construir_modelo(nombre_modelo, semilla, **parametros_modelo),
        particion,
        {clave: valor for clave, valor in evaluacion.items() if clave != "modelo"},
        config_validacion,
    )
    evaluacion.update(resumen_de_validacion(comparacion))
    logger.info(
        "%s -> MAE %.4f  RMSE %.4f  R2 %.4f  spearman %.4f",
        nombre_modelo,
        evaluacion["MAE"],
        evaluacion["RMSE"],
        evaluacion["R2"],
        evaluacion["spearman"],
    )

    evaluaciones = [evaluacion]
    if con_referencias:
        for referencia in ("dummy", "heuristica"):
            if referencia == nombre_modelo:
                continue
            modelo_referencia = entrenar(construir_modelo(referencia, semilla), X_train, y_train)
            evaluaciones.append(evaluar(modelo_referencia, X_test, y_test, referencia))

    curva = None
    if con_curva:
        curva = curva_de_aprendizaje(
            construir_modelo(nombre_modelo, semilla, **parametros_modelo),
            X_train,
            y_train,
            config_validacion.sin_repeticiones(),
        )
        logger.info(
            "Curva de aprendizaje:\n%s",
            curva.round(4).to_string(index=False),
        )
    segmentos = segmentos_debiles(modelo, X_test, y_test)
    debiles = segmentos.loc[segmentos["debil"]] if not segmentos.empty else segmentos
    logger.info(
        "Segmentos del conjunto de prueba: %d analizados, %d con error muy por encima del global",
        len(segmentos),
        len(debiles),
    )

    metricas = construir_informe(evaluaciones)
    guardar_modelo(modelo, rutas.modelo, X_test)
    guardar_metricas(metricas, rutas.metricas)
    _guardar_reportes_de_validacion(rutas.reportes, comparacion, diagnostico, curva, segmentos)
    if con_grafico and curva is not None and rutas.reportes is not None:
        guardar_grafico_curva(curva, rutas.reportes / ARCHIVO_GRAFICO_CURVA)
    return ResultadoEntrenamiento(
        modelo,
        metricas,
        rutas.modelo,
        rutas.metricas,
        chequeos,
        comparacion,
        diagnostico,
        curva,
        segmentos,
    )


def _parsear_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    """Define la interfaz de línea de comandos del script."""
    raiz = buscar_raiz_proyecto()
    parser = argparse.ArgumentParser(
        description="Training pipeline: features de admisiones -> modelo entrenado y metricas."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=raiz / RUTA_FEATURES_RELATIVA,
        help="Parquet de features generado por el feature pipeline (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--modelo",
        choices=sorted(CATALOGO_MODELOS),
        default="extra_trees",
        help="Modelo a entrenar (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--modelo-salida",
        type=Path,
        default=raiz / RUTA_MODELO_RELATIVA,
        help="Ruta del artefacto entrenado (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--metricas",
        type=Path,
        default=raiz / RUTA_METRICAS_RELATIVA,
        help="Ruta de la tabla de metricas (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--chequeos-particion",
        type=Path,
        default=raiz / RUTA_CHEQUEOS_RELATIVA,
        help="Ruta del informe de chequeos de la particion (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--reportes",
        type=Path,
        default=raiz / DIR_REPORTES_RELATIVA,
        help="Directorio de los reportes de validacion (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--validador",
        choices=sorted(VALIDADORES),
        default="repetido",
        help="Esquema de validacion cruzada sobre train (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--pliegues",
        type=int,
        default=PLIEGUES_VALIDACION,
        help="Pliegues de la validacion cruzada (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=REPETICIONES_VALIDACION,
        help="Repeticiones del validador 'repetido' (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--sin-curva",
        action="store_true",
        help="Omite la curva de aprendizaje (mas rapido)",
    )
    parser.add_argument(
        "--grafico",
        action="store_true",
        help="Guarda tambien la curva de aprendizaje como PNG (necesita matplotlib)",
    )
    parser.add_argument(
        "--particion-estricta",
        action="store_true",
        help="Detiene el entrenamiento tambien ante una advertencia de deriva",
    )
    parser.add_argument(
        "--proporcion-test",
        type=float,
        default=PROPORCION_TEST,
        help="Proporcion del conjunto de prueba (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=SEMILLA,
        help="Semilla de la particion y de los modelos (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--presupuesto",
        type=int,
        default=PRESUPUESTO_AUTOML,
        help="Segundos de busqueda, solo con --modelo automl (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--sin-referencias",
        action="store_true",
        help="No mide el dummy ni la heuristica: solo el modelo elegido",
    )
    return parser.parse_args(argumentos)


def main(argumentos: list[str] | None = None) -> int:
    """Punto de entrada del script. Devuelve 0 si el entrenamiento terminó bien, 1 si no."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    opciones = _parsear_argumentos(argumentos)
    parametros = {"presupuesto": opciones.presupuesto} if opciones.modelo == "automl" else {}
    try:
        resultado = ejecutar_pipeline(
            RutasEntrenamiento(
                opciones.features,
                opciones.modelo_salida,
                opciones.metricas,
                opciones.chequeos_particion,
                opciones.reportes,
            ),
            nombre_modelo=opciones.modelo,
            proporcion_test=opciones.proporcion_test,
            semilla=opciones.semilla,
            con_referencias=not opciones.sin_referencias,
            particion_estricta=opciones.particion_estricta,
            configuracion_validacion=ConfiguracionValidacion(
                validador=opciones.validador,
                pliegues=opciones.pliegues,
                repeticiones=opciones.repeticiones,
                semilla=opciones.semilla,
            ),
            con_curva=not opciones.sin_curva,
            con_grafico=opciones.grafico,
            **parametros,
        )
    except (ErrorValidacion, FileNotFoundError) as error:
        # el informe se registra fuera del `except`: un dato o un archivo que faltan no son
        # un fallo del programa, y el traceback seria ruido sobre la informacion util
        informe = str(error)
    else:
        logger.info(
            "Training pipeline terminado:\n%s", resultado.metricas.round(4).to_string(index=False)
        )
        return 0
    logger.error("%s", informe)
    logger.error("No se entreno ningun modelo: no se escribio ningun artefacto")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
