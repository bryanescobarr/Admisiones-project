import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from data.particion import (
    ADVERTENCIA,
    ERROR,
    MAX_SOLAPE,
    OK,
    UMBRAL_AUC_DOMINIO,
    UMBRAL_KS,
    Particion,
    ResultadoChequeo,
    chequear_categorias_nuevas,
    chequear_deriva_del_objetivo,
    chequear_deriva_multivariante,
    chequear_filas_compartidas,
    chequear_indices,
    chequear_nulos,
    chequear_tamanos,
    informe_particion,
    revisar_particion,
    validar_particion_train_test,
)
from data.validacion import ErrorValidacion

FILAS = 200
PROPORCION_TEST = 0.25
SEMILLA = 7
CHEQUEOS_MINIMOS = 10
CORTE_CGPA = 8.3


def _datos(filas: int = FILAS, semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.Series]:
    """Perfiles sintéticos con la forma de las features del proyecto."""
    generador = np.random.default_rng(semilla)
    X = pd.DataFrame(
        {
            "gre_score": generador.integers(290, 341, filas).astype("float64"),
            "toefl_score": generador.integers(95, 121, filas).astype("float64"),
            "university_rating": generador.integers(1, 6, filas).astype("float64"),
            "sop": generador.integers(2, 11, filas) / 2,
            "lor": generador.integers(2, 11, filas) / 2,
            "cgpa": generador.uniform(6.8, 9.9, filas).round(2),
            "research": generador.integers(0, 2, filas).astype("float64"),
        }
    )
    y = pd.Series(generador.uniform(0.35, 0.97, filas).round(3), name="chance_of_admit")
    return X, y


def _particion_valida() -> Particion:
    """Partición aleatoria correcta: la que debe pasar todos los chequeos."""
    X, y = _datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PROPORCION_TEST, random_state=SEMILLA
    )
    return Particion(X_train, X_test, y_train, y_test)


def _severidades(informe: pd.DataFrame, chequeo: str) -> list[str]:
    """Severidades registradas para un chequeo concreto."""
    return list(informe.loc[informe["chequeo"] == chequeo, "severidad"])


# --- Caso valido ------------------------------------------------------------------------


def test_una_particion_aleatoria_pasa_todos_los_chequeos() -> None:
    """El caso normal: partir al azar un dataset homogéneo no genera ni una advertencia."""
    informe = validar_particion_train_test(_particion_valida(), proporcion_esperada=PROPORCION_TEST)

    assert set(informe["severidad"]) == {OK}
    assert len(informe) >= CHEQUEOS_MINIMOS


def test_el_informe_registra_tambien_los_chequeos_que_pasan() -> None:
    """Un informe que solo enumera problemas no dice qué se comprobó."""
    informe = validar_particion_train_test(_particion_valida())

    assert list(informe.columns) == [
        "chequeo",
        "columna",
        "severidad",
        "estadistico",
        "umbral",
        "detalle",
    ]
    assert "deriva_multivariante" in informe["chequeo"].tolist()


# --- Fuga de informacion: error controlado ----------------------------------------------


def test_los_indices_solapados_son_un_error_que_detiene_el_proceso() -> None:
    """Una fila en los dos conjuntos invalida la evaluación: no es una advertencia."""
    X, y = _datos()
    particion = Particion(X.iloc[:150], X.iloc[100:], y.iloc[:150], y.iloc[100:])

    with pytest.raises(ErrorValidacion, match="indices_solapados"):
        validar_particion_train_test(particion)


def test_repetir_perfiles_en_prueba_es_fuga_de_informacion() -> None:
    """Mismo perfil con otro índice: el modelo lo memorizó y acertará sin aprender."""
    X, y = _datos()
    X_train, y_train = X.iloc[:150], y.iloc[:150]
    X_test = X.iloc[:50].reset_index(drop=True).set_index(np.arange(1000, 1050))
    y_test = pd.Series(y.iloc[:50].to_numpy(), index=X_test.index, name=y.name)

    with pytest.raises(ErrorValidacion, match="filas_compartidas"):
        validar_particion_train_test(Particion(X_train, X_test, y_train, y_test))


def test_un_solape_pequeno_es_solo_una_advertencia() -> None:
    """Por debajo del 5 % se avisa, pero no se bloquea: es el umbral de deepchecks."""
    X, _ = _datos()
    X_train = X.iloc[:150]
    X_test = X.iloc[150:].copy()
    X_test.iloc[0] = X_train.iloc[0]

    resultado = chequear_filas_compartidas(X_train, X_test)

    assert resultado.severidad == ADVERTENCIA
    assert resultado.estadistico is not None
    assert resultado.estadistico < MAX_SOLAPE


def test_sin_solape_el_chequeo_de_indices_pasa() -> None:
    """El caso correcto, para que la prueba anterior signifique algo."""
    particion = _particion_valida()

    assert chequear_indices(particion.X_train, particion.X_test).severidad == OK


# --- Representatividad: advertencias ----------------------------------------------------


def test_una_proporcion_distinta_de_la_pedida_se_advierte() -> None:
    """Si se pidió 25 % de prueba y salió 50 %, algo hizo la partición mal."""
    X, _ = _datos()
    resultados = chequear_tamanos(X.iloc[:100], X.iloc[100:], PROPORCION_TEST)

    severidades = {resultado.chequeo: resultado.severidad for resultado in resultados}
    assert severidades["proporcion_de_la_particion"] == ADVERTENCIA


def test_un_conjunto_de_prueba_diminuto_se_advierte() -> None:
    """Con 5 filas, cualquier métrica tiene un intervalo enorme."""
    X, _ = _datos()
    resultados = chequear_tamanos(X.iloc[:195], X.iloc[195:], 0.025)

    severidades = {resultado.chequeo: resultado.severidad for resultado in resultados}
    assert severidades["tamano_del_conjunto_de_prueba"] == ADVERTENCIA


def test_un_objetivo_desplazado_en_prueba_se_advierte() -> None:
    """Con el objetivo corrido, hasta un modelo perfecto parecería sesgado."""
    _, y = _datos()
    y_train = y.iloc[:150]
    y_test = y.iloc[150:] * 0.4

    resultado = chequear_deriva_del_objetivo(y_train, y_test)

    assert resultado.severidad == ADVERTENCIA
    assert resultado.estadistico is not None and resultado.estadistico > UMBRAL_KS


def test_una_variable_con_otra_distribucion_se_advierte() -> None:
    """El conjunto de prueba solo tiene perfiles de CGPA alto: no representa el problema."""
    X, y = _datos()
    X_train, y_train = X.iloc[:150], y.iloc[:150]
    X_test, y_test = X.iloc[150:].copy(), y.iloc[150:]
    X_test["cgpa"] = 9.8

    informe = validar_particion_train_test(Particion(X_train, X_test, y_train, y_test))

    assert ADVERTENCIA in _severidades(informe, "deriva_de_la_variable")


def test_una_categoria_que_solo_aparece_en_prueba_se_advierte() -> None:
    """El modelo nunca vio esa categoría: no puede haber aprendido nada de ella."""
    X, _ = _datos()
    X_train = X.iloc[:150].copy()
    X_train["university_rating"] = X_train["university_rating"].clip(upper=4)
    X_test = X.iloc[150:].copy()
    X_test["university_rating"] = 5.0

    resultados = chequear_categorias_nuevas(X_train, X_test, ["university_rating"])

    assert resultados[0].severidad == ADVERTENCIA
    assert "5.0" in resultados[0].detalle


def test_los_nulos_repartidos_de_forma_desigual_se_advierten() -> None:
    """Evaluar sobre perfiles mucho más incompletos de los que se aprendió a tratar."""
    X, _ = _datos()
    X_train = X.iloc[:150]
    X_test = X.iloc[150:].copy()
    X_test.loc[X_test.index[:30], "toefl_score"] = np.nan

    resultados = chequear_nulos(X_train, X_test)

    assert [resultado.columna for resultado in resultados] == ["toefl_score"]
    assert resultados[0].severidad == ADVERTENCIA


def test_los_nulos_repartidos_por_igual_no_generan_ruido() -> None:
    """El chequeo no debe disparar por una diferencia de un punto porcentual."""
    particion = _particion_valida()

    assert chequear_nulos(particion.X_train, particion.X_test) == []


def test_el_clasificador_de_dominio_detecta_conjuntos_distinguibles() -> None:
    """Deriva conjunta: cada variable puede parecer igual y las combinaciones no serlo."""
    X, _ = _datos(400)
    X_train = X[X["cgpa"] < CORTE_CGPA]
    X_test = X[X["cgpa"] >= CORTE_CGPA]

    resultado = chequear_deriva_multivariante(X_train, X_test, SEMILLA)

    assert resultado.severidad == ADVERTENCIA
    assert resultado.estadistico is not None and resultado.estadistico > UMBRAL_AUC_DOMINIO


def test_el_clasificador_de_dominio_no_distingue_una_particion_aleatoria() -> None:
    """El caso bueno: AUC cerca de 0.5, los conjuntos son indistinguibles."""
    particion = _particion_valida()

    resultado = chequear_deriva_multivariante(particion.X_train, particion.X_test, SEMILLA)

    assert resultado.severidad == OK
    assert resultado.estadistico is not None and resultado.estadistico < UMBRAL_AUC_DOMINIO


# --- Modo estricto y utilidades ---------------------------------------------------------


def test_en_modo_estricto_una_advertencia_detiene_el_proceso() -> None:
    """Para quien prefiera no entrenar con una partición sospechosa."""
    X, y = _datos()
    X_train, y_train = X.iloc[:150], y.iloc[:150]
    X_test, y_test = X.iloc[150:].copy(), y.iloc[150:]
    X_test["cgpa"] = 9.8

    particion = Particion(X_train, X_test, y_train, y_test)
    validar_particion_train_test(particion)  # sin estricto: no rompe

    with pytest.raises(ErrorValidacion, match="deriva_de_la_variable"):
        validar_particion_train_test(particion, estricto=True)


def test_revisar_particion_no_lanza_excepciones() -> None:
    """La versión de inspección: devuelve los resultados aunque haya fuga."""
    X, y = _datos()
    resultados = revisar_particion(
        Particion(X.iloc[:150], X.iloc[100:], y.iloc[:150], y.iloc[100:])
    )

    assert any(resultado.severidad == ERROR for resultado in resultados)
    assert not informe_particion(resultados).empty


def test_el_texto_de_un_chequeo_es_legible() -> None:
    """El mensaje se lee en un log, así que tiene que bastarse solo."""
    resultado = ResultadoChequeo("deriva_de_la_variable", "cgpa", ADVERTENCIA, "detalle", 0.31, 0.2)

    assert str(resultado) == "[cgpa] deriva_de_la_variable: detalle (0.3100 vs. umbral 0.2)"
