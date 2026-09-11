import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from model.validacion import (
    DEGRADACION_MAXIMA,
    METRICAS_REGRESION,
    ConfiguracionValidacion,
    EstratificadoPorCuantiles,
    comparar_train_cv_test,
    construir_validador,
    curva_de_aprendizaje,
    diagnosticar_generalizacion,
    guardar_grafico_curva,
    segmentos_debiles,
    validar_con_cruzada,
)

FILAS = 120
PLIEGUES = 3
SEMILLA = 11
TOLERANCIA = 1e-9
# diferencia maxima admitida entre las medias del objetivo de cada pliegue estratificado
DIFERENCIA_ENTRE_PLIEGUES = 0.05
TALLAS_DE_PRUEBA = 3
# corte del tercio de CGPA en el que falla el modelo de juguete
CORTE_CGPA_BAJO = 7.8


def _datos(filas: int = FILAS) -> tuple[pd.DataFrame, pd.Series]:
    """Un problema de regresión fácil: señal clara más un poco de ruido."""
    generador = np.random.default_rng(SEMILLA)
    X = pd.DataFrame(
        {
            "cgpa": generador.uniform(6.8, 9.9, filas).round(2),
            "gre_score": generador.integers(290, 341, filas).astype("float64"),
            "research": generador.integers(0, 2, filas).astype("float64"),
        }
    )
    y = pd.Series(
        (0.5 + 0.05 * (X["cgpa"] - 6.8) + 0.002 * (X["gre_score"] - 290)).round(4)
        + generador.normal(0, 0.01, filas),
        name="chance_of_admit",
    )
    return X, y


def _configuracion(validador: str = "kfold") -> ConfiguracionValidacion:
    """Configuración pequeña para que las pruebas sean rápidas y deterministas."""
    return ConfiguracionValidacion(validador, pliegues=PLIEGUES, repeticiones=2, semilla=SEMILLA)


def _comparacion(**valores: float) -> pd.DataFrame:
    """Tabla de comparación sintética, para probar el diagnóstico sin entrenar nada."""
    base = {
        "entrenamiento": 0.040,
        "validacion_cruzada": 0.045,
        "desviacion_cv": 0.004,
        "prueba": 0.046,
        "r2_cv": 0.80,
    }
    base.update(valores)
    return pd.DataFrame(
        [
            {
                "metrica": "MAE",
                "entrenamiento": base["entrenamiento"],
                "validacion_cruzada": base["validacion_cruzada"],
                "desviacion_cv": base["desviacion_cv"],
                "prueba": base["prueba"],
                "brecha_train_cv": base["validacion_cruzada"] - base["entrenamiento"],
            },
            {
                "metrica": "R2",
                "entrenamiento": 0.9,
                "validacion_cruzada": base["r2_cv"],
                "desviacion_cv": 0.03,
                "prueba": base["r2_cv"],
                "brecha_train_cv": base["r2_cv"] - 0.9,
            },
        ]
    )


# --- Esquemas de validacion cruzada -----------------------------------------------------


@pytest.mark.parametrize(
    ("nombre", "tipo"),
    [
        ("kfold", KFold),
        ("repetido", RepeatedKFold),
        ("estratificado", EstratificadoPorCuantiles),
        ("series_temporales", TimeSeriesSplit),
    ],
)
def test_el_catalogo_de_validadores_devuelve_el_esquema_pedido(nombre: str, tipo: type) -> None:
    """Cada nombre del catálogo construye el validador que le corresponde."""
    assert isinstance(construir_validador(nombre, pliegues=PLIEGUES, semilla=SEMILLA), tipo)


def test_un_validador_desconocido_enumera_las_opciones() -> None:
    """El mensaje dice qué se puede usar en vez de dejar un error opaco."""
    with pytest.raises(ValueError, match="Disponibles"):
        construir_validador("validacion_magica")


def test_el_estratificado_reparte_el_objetivo_por_igual_entre_pliegues() -> None:
    """Con el objetivo ordenado, un KFold dejaría pliegues sesgados y este no."""
    X, _ = _datos()
    y = pd.Series(np.linspace(0.3, 0.99, len(X)), name="chance_of_admit")
    validador = EstratificadoPorCuantiles(n_splits=PLIEGUES, semilla=SEMILLA)

    medias = [float(y.iloc[prueba].mean()) for _, prueba in validador.split(X, y)]

    assert validador.get_n_splits() == PLIEGUES
    assert max(medias) - min(medias) < DIFERENCIA_ENTRE_PLIEGUES


def test_los_pliegues_del_estratificado_no_se_solapan() -> None:
    """Cada fila valida exactamente una vez: si no, la media no significa nada."""
    X, y = _datos()
    validador = EstratificadoPorCuantiles(n_splits=PLIEGUES, semilla=SEMILLA)

    validaciones = [conjunto for _, conjunto in validador.split(X, y)]
    todas = np.concatenate(validaciones)

    assert len(todas) == len(np.unique(todas)) == len(X)


# --- Validacion cruzada -----------------------------------------------------------------


def test_la_validacion_cruzada_devuelve_una_fila_por_metrica() -> None:
    """Las cinco métricas de regresión, medidas en entrenamiento y en validación."""
    X, y = _datos()

    validacion = validar_con_cruzada(LinearRegression(), X, y, _configuracion())

    assert validacion["metrica"].tolist() == list(METRICAS_REGRESION)
    assert {"entrenamiento", "validacion_cruzada", "desviacion_cv", "brecha_train_cv"} <= set(
        validacion.columns
    )


def test_los_errores_se_reportan_en_positivo() -> None:
    """scikit-learn los devuelve como `neg_*`; una tabla que se lee no puede hacerlo."""
    X, y = _datos()

    validacion = validar_con_cruzada(LinearRegression(), X, y, _configuracion())
    mae = validacion.loc[validacion["metrica"] == "MAE"].iloc[0]

    assert mae["validacion_cruzada"] > 0
    assert mae["entrenamiento"] > 0


def test_la_validacion_registra_como_se_ejecuto() -> None:
    """El artefacto tiene que decir con qué validador y semilla se produjo."""
    X, y = _datos()

    validacion = validar_con_cruzada(LinearRegression(), X, y, _configuracion("repetido"))

    assert set(validacion["validador"]) == {"repetido"}
    assert set(validacion["semilla"]) == {SEMILLA}
    assert set(validacion["repeticiones"]) == {2}


def test_un_arbol_sin_podar_muestra_la_brecha_y_un_lineal_no() -> None:
    """La prueba de que la brecha mide lo que dice: memorizar frente a generalizar."""
    X, y = _datos()

    arbol = validar_con_cruzada(DecisionTreeRegressor(random_state=SEMILLA), X, y, _configuracion())
    lineal = validar_con_cruzada(LinearRegression(), X, y, _configuracion())

    brecha_arbol = arbol.loc[arbol["metrica"] == "MAE", "brecha_train_cv"].iloc[0]
    brecha_lineal = lineal.loc[lineal["metrica"] == "MAE", "brecha_train_cv"].iloc[0]
    assert brecha_arbol > brecha_lineal


def test_un_modelo_constante_no_rompe_la_correlacion_de_rangos() -> None:
    """El dummy no ordena nada: spearman queda nula en vez de reventar la validación."""
    X, y = _datos()

    validacion = validar_con_cruzada(DummyRegressor(strategy="mean"), X, y, _configuracion())

    assert np.isnan(validacion.loc[validacion["metrica"] == "spearman", "validacion_cruzada"]).all()


def test_la_configuracion_de_la_curva_quita_las_repeticiones() -> None:
    """La curva ya ajusta una vez por talla y pliegue: repetirlo no cambia su forma."""
    assert _configuracion("repetido").sin_repeticiones().validador == "kfold"
    assert _configuracion("estratificado").sin_repeticiones().validador == "estratificado"


# --- Comparacion train / cv / test ------------------------------------------------------


def test_la_comparacion_reune_los_tres_escenarios() -> None:
    """La tabla del entregable: entrenamiento, validación cruzada y prueba, lado a lado."""
    X, y = _datos()
    validacion = validar_con_cruzada(LinearRegression(), X, y, _configuracion())

    comparacion = comparar_train_cv_test(validacion, {"MAE": 0.05, "R2": 0.8})
    mae = comparacion.loc[comparacion["metrica"] == "MAE"].iloc[0]

    assert mae["prueba"] == pytest.approx(0.05)
    assert mae["diferencia_cv_prueba"] == pytest.approx(
        0.05 - mae["validacion_cruzada"], abs=TOLERANCIA
    )


# --- Diagnostico ------------------------------------------------------------------------


def test_diagnostica_sobreajuste_cuando_el_modelo_memoriza() -> None:
    """MAE 0 en entrenamiento y 0.045 en validación: eso es memoria, no aprendizaje."""
    diagnostico = diagnosticar_generalizacion(_comparacion(entrenamiento=0.0))
    fila = diagnostico.loc[diagnostico["aspecto"] == "sobreajuste"].iloc[0]

    assert fila["veredicto"] == "sobreajuste"
    assert "limitar la capacidad" in fila["accion_recomendada"]


def test_diagnostica_ajuste_adecuado_cuando_la_brecha_es_pequena() -> None:
    """El caso bueno: el modelo rinde casi igual dentro y fuera."""
    diagnostico = diagnosticar_generalizacion(_comparacion(entrenamiento=0.044))
    fila = diagnostico.loc[diagnostico["aspecto"] == "sobreajuste"].iloc[0]

    assert fila["veredicto"] == "ajuste adecuado"


def test_diagnostica_subajuste_con_un_r2_bajo() -> None:
    """Si no explica ni la mitad de la varianza, el problema es de capacidad."""
    diagnostico = diagnosticar_generalizacion(_comparacion(r2_cv=0.2))
    fila = diagnostico.loc[diagnostico["aspecto"] == "subajuste"].iloc[0]

    assert fila["veredicto"] == "subajuste"
    assert "capacidad" in fila["accion_recomendada"]


def test_diagnostica_inestabilidad_cuando_los_pliegues_discrepan() -> None:
    """Con esta varianza, comparar modelos por décimas no significa nada."""
    diagnostico = diagnosticar_generalizacion(_comparacion(desviacion_cv=0.03))
    fila = diagnostico.loc[diagnostico["aspecto"] == "estabilidad"].iloc[0]

    assert fila["veredicto"] == "inestable"


def test_diagnostica_la_degradacion_entre_validacion_y_prueba() -> None:
    """Si la prueba se aparta de la validación, el número de prueba es suerte."""
    diagnostico = diagnosticar_generalizacion(_comparacion(prueba=0.070))
    fila = diagnostico.loc[diagnostico["aspecto"] == "degradacion_cv_prueba"].iloc[0]

    assert fila["veredicto"] == "la prueba no confirma la validacion"
    assert f"{DEGRADACION_MAXIMA:.0%}" in fila["evidencia"]


def test_el_diagnostico_cubre_los_cuatro_aspectos() -> None:
    """Ningún aspecto se queda sin veredicto ni sin acción recomendada."""
    diagnostico = diagnosticar_generalizacion(_comparacion())

    assert diagnostico["aspecto"].tolist() == [
        "sobreajuste",
        "subajuste",
        "estabilidad",
        "degradacion_cv_prueba",
    ]
    assert diagnostico["accion_recomendada"].str.len().min() > 0


# --- Curva de aprendizaje ---------------------------------------------------------------


def test_la_curva_de_aprendizaje_mide_el_error_por_tamano() -> None:
    """Con más datos, el error de validación baja: la curva tiene que verlo."""
    X, y = _datos()

    curva = curva_de_aprendizaje(LinearRegression(), X, y, _configuracion(), tallas=(0.2, 0.6, 1.0))

    assert list(curva.columns) == [
        "muestras_entrenamiento",
        "MAE_entrenamiento",
        "MAE_validacion",
        "desviacion_validacion",
    ]
    assert len(curva) == TALLAS_DE_PRUEBA
    assert curva["MAE_validacion"].iloc[-1] < curva["MAE_validacion"].iloc[0]


def test_el_grafico_de_la_curva_se_guarda_si_hay_matplotlib(tmp_path: Path) -> None:
    """La visualización del entregable, cuando la librería de dibujo está disponible."""
    X, y = _datos()
    curva = curva_de_aprendizaje(LinearRegression(), X, y, _configuracion(), tallas=(0.5, 1.0))
    ruta = tmp_path / "graficos" / "curva.png"

    assert guardar_grafico_curva(curva, ruta) == ruta
    assert ruta.stat().st_size > 0


def test_sin_matplotlib_se_avisa_y_el_pipeline_sigue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un pipeline que se cae porque falta una librería de dibujo sería un mal diseño."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    curva = pd.DataFrame(
        {
            "muestras_entrenamiento": [10, 20],
            "MAE_entrenamiento": [0.05, 0.04],
            "MAE_validacion": [0.06, 0.05],
            "desviacion_validacion": [0.01, 0.01],
        }
    )
    ruta = tmp_path / "curva.png"

    assert guardar_grafico_curva(curva, ruta) is None
    assert not ruta.exists()


# --- Segmentos debiles ------------------------------------------------------------------


class _ModeloQueFallaConCgpaBajo:
    """Modelo de juguete que se equivoca solo con una parte de los aspirantes."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice bien salvo en el tercio de CGPA más bajo."""
        return np.where(X["cgpa"] < CORTE_CGPA_BAJO, 0.99, 0.70)


def test_detecta_el_segmento_en_el_que_el_modelo_falla() -> None:
    """El error medio esconde a quién perjudica; el análisis por segmento lo saca."""
    X, _ = _datos()
    y = pd.Series(np.full(len(X), 0.70), name="chance_of_admit")

    segmentos = segmentos_debiles(_ModeloQueFallaConCgpaBajo(), X, y, columnas=["cgpa"])

    assert segmentos["debil"].any()
    assert segmentos.iloc[0]["atributo"] == "cgpa"
    assert segmentos.iloc[0]["razon_vs_global"] > 1


def test_un_modelo_uniforme_no_tiene_segmentos_debiles() -> None:
    """Si el error se reparte por igual, ningún tercio queda marcado."""
    X, y = _datos()
    modelo = LinearRegression().fit(X, y)

    segmentos = segmentos_debiles(modelo, X, y, columnas=["cgpa", "gre_score"])

    assert not segmentos["debil"].any()
    assert set(segmentos["segmento"]) == {"tercio 1", "tercio 2", "tercio 3"}


def test_las_columnas_constantes_se_omiten() -> None:
    """No se puede partir en tercios una columna con un solo valor."""
    X, y = _datos()
    X = X.assign(constante=1.0)
    modelo = LinearRegression().fit(X, y)

    segmentos = segmentos_debiles(modelo, X, y)

    assert "constante" not in set(segmentos["atributo"])
