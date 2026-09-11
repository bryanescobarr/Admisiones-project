import pandas as pd
import pytest

from data.validacion import (
    MIN_FILAS_PARA_PROPORCION,
    ErrorValidacion,
    EsquemaDatos,
    ReglaColumna,
    ReglaRelacion,
    Violacion,
    comparar_datasets,
    revisar,
    validar,
)

# constantes de las comparaciones, para no repetir numeros sueltos en los asserts
NULOS_ESPERADOS = 10
FILAS_DE_MUESTRA_PEQUENA = 3
VIOLACIONES_ESPERADAS = 2
TOLERANCIA = 1e-9

ESQUEMA_MINIMO = EsquemaDatos(
    columnas={
        "puntaje": ReglaColumna(tipos=("float64",), rango=(0, 10), max_nulos=0.5),
        "nivel": ReglaColumna(tipos=("float64",), categorias=(1.0, 2.0, 3.0)),
    }
)


def _datos_validos() -> pd.DataFrame:
    """Dos filas que cumplen `ESQUEMA_MINIMO`."""
    return pd.DataFrame({"puntaje": [8.0, 4.5], "nivel": [1.0, 3.0]})


# --- Estructura del conjunto de datos ---------------------------------------------------


def test_validar_acepta_los_datos_correctos() -> None:
    """Un conjunto que cumple el contrato se devuelve intacto."""
    datos = _datos_validos()

    assert validar(datos, ESQUEMA_MINIMO) is datos


def test_detecta_una_columna_obligatoria_que_falta() -> None:
    """Si la fuente deja de enviar una columna, el contrato se rompe."""
    violaciones = revisar(pd.DataFrame({"puntaje": [8.0]}), ESQUEMA_MINIMO)

    assert [violacion.regla for violacion in violaciones] == ["columna_faltante"]


def test_una_columna_opcional_que_falta_no_es_un_problema() -> None:
    """Las columnas declaradas `obligatoria=False` solo se validan si están."""
    esquema = EsquemaDatos(
        columnas={
            "puntaje": ReglaColumna(tipos=("float64",)),
            "extra": ReglaColumna(tipos=("float64",), obligatoria=False),
        }
    )

    assert revisar(pd.DataFrame({"puntaje": [8.0]}), esquema) == []


def test_detecta_una_columna_inesperada() -> None:
    """Una columna que nadie declaró suele significar que la fuente cambió sin avisar."""
    datos = _datos_validos().assign(sorpresa=[1, 2])

    violaciones = revisar(datos, ESQUEMA_MINIMO)

    assert [violacion.regla for violacion in violaciones] == ["columna_inesperada"]


def test_las_columnas_extra_se_permiten_si_el_esquema_lo_dice() -> None:
    """El mismo caso, con el contrato configurado para tolerarlas."""
    esquema = EsquemaDatos(columnas=dict(ESQUEMA_MINIMO.columnas), permitir_columnas_extra=True)

    assert revisar(_datos_validos().assign(sorpresa=[1, 2]), esquema) == []


# --- Reglas por columna -----------------------------------------------------------------


def test_detecta_un_tipo_inesperado() -> None:
    """`puntaje` como texto no sirve aunque sus valores parezcan correctos."""
    datos = _datos_validos().astype({"puntaje": "string"})

    violaciones = revisar(datos, ESQUEMA_MINIMO)

    assert [violacion.regla for violacion in violaciones] == ["tipo_inesperado"]


def test_detecta_un_valor_fuera_de_rango() -> None:
    """El detalle incluye los valores culpables, que es lo que se va a mirar primero."""
    datos = _datos_validos()
    datos.loc[0, "puntaje"] = 42.0

    violaciones = revisar(datos, ESQUEMA_MINIMO)

    assert violaciones[0].regla == "fuera_de_rango"
    assert "42.0" in violaciones[0].detalle
    assert violaciones[0].filas_afectadas == 1


def test_detecta_una_categoria_invalida() -> None:
    """Un nivel 7 en un conjunto declarado de 1 a 3 es un error de la fuente."""
    datos = _datos_validos()
    datos.loc[1, "nivel"] = 7.0

    violaciones = revisar(datos, ESQUEMA_MINIMO)

    assert violaciones[0].regla == "categoria_invalida"
    assert "7.0" in violaciones[0].detalle


def test_detecta_el_exceso_de_nulos_en_un_conjunto_grande() -> None:
    """Con datos suficientes, la proporción de nulos sí es informativa."""
    esquema = EsquemaDatos(columnas={"puntaje": ReglaColumna(max_nulos=0.10)})
    valores = [None] * 10 + [1.0] * 40  # 20 % de nulos en 50 filas

    violaciones = revisar(pd.DataFrame({"puntaje": valores}), esquema)

    assert violaciones[0].regla == "exceso_de_nulos"
    assert violaciones[0].filas_afectadas == NULOS_ESPERADOS


def test_el_umbral_relativo_no_se_aplica_a_muestras_pequenas() -> None:
    """En tres filas, un nulo ya es un 33 %: la proporción no dice nada todavía."""
    esquema = EsquemaDatos(columnas={"puntaje": ReglaColumna(max_nulos=0.10)})

    assert revisar(pd.DataFrame({"puntaje": [None, 1.0, 2.0]}), esquema) == []
    assert MIN_FILAS_PARA_PROPORCION > FILAS_DE_MUESTRA_PEQUENA


def test_un_maximo_de_cero_nulos_se_exige_siempre() -> None:
    """`max_nulos=0` es una regla absoluta: se aplica aunque haya dos filas."""
    esquema = EsquemaDatos(columnas={"objetivo": ReglaColumna(max_nulos=0.0)})

    violaciones = revisar(pd.DataFrame({"objetivo": [None, 1.0]}), esquema)

    assert violaciones[0].regla == "exceso_de_nulos"


def test_valida_el_formato_de_fecha() -> None:
    """Hoy ninguna columna del proyecto es temporal, pero el contrato sabe comprobarlas."""
    esquema = EsquemaDatos(columnas={"fecha": ReglaColumna(formato_fecha="%Y-%m-%d")})

    assert revisar(pd.DataFrame({"fecha": ["2026-09-11", None]}), esquema) == []

    violaciones = revisar(pd.DataFrame({"fecha": ["11/09/2026"]}), esquema)
    assert violaciones[0].regla == "formato_de_fecha_invalido"


def test_detecta_una_clave_duplicada() -> None:
    """Una columna declarada única no admite repetidos."""
    esquema = EsquemaDatos(columnas={"id": ReglaColumna(unica=True)})

    assert revisar(pd.DataFrame({"id": [1, 2, 3]}), esquema) == []

    violaciones = revisar(pd.DataFrame({"id": [1, 2, 2]}), esquema)
    assert violaciones[0].regla == "clave_duplicada"


# --- Integridad entre registros ---------------------------------------------------------


def test_detecta_registros_duplicados() -> None:
    """Sin identificador, dos filas idénticas son indistinguibles y sobran."""
    esquema = EsquemaDatos(columnas=dict(ESQUEMA_MINIMO.columnas), filas_unicas=True)
    datos = pd.DataFrame({"puntaje": [8.0, 8.0], "nivel": [1.0, 1.0]})

    violaciones = revisar(datos, esquema)

    assert violaciones[0].regla == "fila_duplicada"
    assert violaciones[0].filas_afectadas == 1


def test_detecta_una_fila_sin_informacion_suficiente() -> None:
    """Un registro casi vacío no se puede usar para entrenar ni para predecir."""
    esquema = EsquemaDatos(columnas=dict(ESQUEMA_MINIMO.columnas), min_valores_por_fila=2)
    datos = pd.DataFrame({"puntaje": [8.0, None], "nivel": [1.0, None]})

    violaciones = revisar(datos, esquema)

    assert violaciones[0].regla == "fila_sin_informacion"
    assert violaciones[0].filas_afectadas == 1


# --- Integridad entre campos ------------------------------------------------------------


RELACION_SUMA = ReglaRelacion(
    nombre="total_coherente",
    descripcion="total debe ser la suma de parte_a y parte_b",
    columnas=("parte_a", "parte_b", "total"),
    condicion=lambda datos: (
        (datos["parte_a"] + datos["parte_b"] - datos["total"]).abs() < TOLERANCIA
    ),
)
ESQUEMA_RELACION = EsquemaDatos(
    columnas={
        "parte_a": ReglaColumna(),
        "parte_b": ReglaColumna(),
        "total": ReglaColumna(obligatoria=False),
    },
    relaciones=(RELACION_SUMA,),
)


def test_una_relacion_entre_campos_se_cumple() -> None:
    """Datos coherentes entre sí no generan ninguna violación."""
    datos = pd.DataFrame({"parte_a": [1.0], "parte_b": [2.0], "total": [3.0]})

    assert revisar(datos, ESQUEMA_RELACION) == []


def test_detecta_una_relacion_entre_campos_incumplida() -> None:
    """Si alguien cambia una fórmula y olvida la otra mitad, se ve aquí."""
    datos = pd.DataFrame({"parte_a": [1.0, 1.0], "parte_b": [2.0, 2.0], "total": [3.0, 99.0]})

    violaciones = revisar(datos, ESQUEMA_RELACION)

    assert violaciones[0].regla == "relacion_incumplida"
    assert violaciones[0].filas_afectadas == 1


def test_una_relacion_se_omite_si_falta_alguna_de_sus_columnas() -> None:
    """Las relaciones sobre atributos opcionales no estorban cuando no se generan."""
    datos = pd.DataFrame({"parte_a": [1.0], "parte_b": [2.0]})

    assert revisar(datos, ESQUEMA_RELACION) == []


# --- Informe y excepción ----------------------------------------------------------------


def test_el_informe_recoge_todas_las_violaciones_a_la_vez() -> None:
    """Fallar en la primera obliga a ejecutar el pipeline una vez por cada problema."""
    datos = pd.DataFrame({"puntaje": [42.0], "nivel": [7.0]})

    with pytest.raises(ErrorValidacion) as error:
        validar(datos, ESQUEMA_MINIMO, contexto="las pruebas")

    assert len(error.value.violaciones) == VIOLACIONES_ESPERADAS
    mensaje = str(error.value)
    assert "las pruebas" in mensaje
    assert "fuera_de_rango" in mensaje
    assert "categoria_invalida" in mensaje


def test_el_texto_de_una_violacion_es_legible() -> None:
    """El mensaje se lee en un log, así que tiene que bastarse solo."""
    violacion = Violacion("fuera_de_rango", "cgpa", "valores fuera de [0, 10]", 3)

    assert str(violacion) == "[cgpa] fuera_de_rango: valores fuera de [0, 10] (3 filas)"
    assert "1 fila)" in str(Violacion("fuera_de_rango", "cgpa", "detalle", 1))


# --- Integridad entre datasets ----------------------------------------------------------


def test_comparar_datasets_acepta_una_transformacion_limpia() -> None:
    """Copiar columnas y quitar filas es legítimo; inventar datos no."""
    origen = pd.DataFrame({"a": [1.0, 2.0, 2.0], "b": [10.0, 20.0, 20.0]})
    destino = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})

    assert comparar_datasets(origen, destino, columnas_conservadas=["a", "b"]) is destino


def test_comparar_datasets_detecta_filas_inventadas() -> None:
    """Esta etapa limpia y deriva: nunca puede devolver más registros de los que recibió."""
    origen = pd.DataFrame({"a": [1.0]})
    destino = pd.DataFrame({"a": [1.0, 1.0]})

    with pytest.raises(ErrorValidacion, match="filas_inventadas"):
        comparar_datasets(origen, destino, columnas_conservadas=["a"])


def test_comparar_datasets_detecta_una_columna_perdida() -> None:
    """Una columna que debía conservarse y no llegó a la salida."""
    origen = pd.DataFrame({"a": [1.0], "b": [2.0]})
    destino = pd.DataFrame({"a": [1.0]})

    with pytest.raises(ErrorValidacion, match="columna_perdida"):
        comparar_datasets(origen, destino, columnas_conservadas=["a", "b"])


def test_comparar_datasets_detecta_un_valor_que_no_estaba_en_el_origen() -> None:
    """El fallo silencioso más caro: alterar una columna que se creía solo copiar."""
    origen = pd.DataFrame({"a": [1.0, 2.0]})
    destino = pd.DataFrame({"a": [1.0, 99.0]})

    with pytest.raises(ErrorValidacion, match="valor_no_presente_en_el_origen"):
        comparar_datasets(origen, destino, columnas_conservadas=["a"])


def test_comparar_datasets_detecta_nulos_introducidos() -> None:
    """Una conversión que convierte valores en nulos sin decirlo."""
    origen = pd.DataFrame({"a": [1.0, 2.0]})
    destino = pd.DataFrame({"a": [1.0, None]})

    with pytest.raises(ErrorValidacion, match="nulos_introducidos"):
        comparar_datasets(origen, destino, columnas_conservadas=["a"])
