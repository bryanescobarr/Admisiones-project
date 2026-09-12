"""Feature pipeline del proyecto de admisiones: de datos crudos a features en disco.

Automatiza, como script ejecutable, lo que hasta ahora solo existía dentro de los
notebooks `2-exploration` (limpieza y tipado) y `4-feat_eng` (dominio y atributos
derivados). Sigue la arquitectura FTI descrita en `src/README.md`: esta etapa lee la
fuente cruda, produce features y las guarda; el entrenamiento es otra etapa.

    data/01_raw/Admission_Predict.csv  ->  data/04_feature/admisiones_features.parquet

**Lo que el pipeline hace y lo que deliberadamente no hace.** Aquí solo ocurren
transformaciones que dependen de una fila a la vez o de reglas documentadas: normalizar
nombres, unificar los centinelas de nulo, tipar, eliminar duplicados exactos, recortar al
dominio de `data/01_raw/Informacion.txt` y —opcionalmente— añadir los atributos derivados
de `4-feat_eng`. La imputación y el escalado **no** se hacen aquí: sus parámetros (la
mediana, la media, la desviación) se aprenden de los datos y, calculados sobre el dataset
completo, filtrarían información del conjunto de prueba al de entrenamiento. Esos pasos
viven en el `Pipeline` de scikit-learn que ajusta el training pipeline, que es quien
particiona.

**Validación antes de persistir.** El pipeline tiene tres puertas de calidad, todas
anteriores a la escritura del parquet: `ESQUEMA_FUENTE` sobre los datos tipados,
`ESQUEMA_FEATURES` sobre lo que se va a guardar y una comparación de integridad entre
ambos conjuntos. Los contratos son declarativos y viven en este módulo, junto a las
constantes que los justifican; el motor que los aplica está en `src/data/validacion.py`.
Si alguna regla falla, el script escribe un informe con **todas** las violaciones, termina
con código de salida 1 y **no toca el archivo de features**.

Uso:

    uv run python src/pipelines/feature_pipeline/feature_pipeline.py
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py --con-derivados
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py \
        --entrada otros/datos.csv --salida otros/features.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# el script se ejecuta directamente (`python src/pipelines/.../feature_pipeline.py`), asi
# que `src/` no esta en sys.path: hay que anadirlo antes de importar `data.transformaciones`
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.transformaciones import (  # noqa: E402
    a_tipos_sklearn,
    agregar_atributos_derivados,
    recortar_a_dominio,
)
from data.validacion import (  # noqa: E402
    ErrorValidacion,
    EsquemaDatos,
    ReglaColumna,
    ReglaRelacion,
    Violacion,
    comparar_datasets,
    validar,
)

logger = logging.getLogger(__name__)

TARGET = "chance_of_admit"

# nombres esperados tras normalizar; el orden es el del CSV original
COLUMNAS_ESPERADAS: list[str] = [
    "gre_score",
    "toefl_score",
    "university_rating",
    "sop",
    "lor",
    "cgpa",
    "research",
    TARGET,
]

# representaciones de nulo encontradas o previsibles en la fuente (auditadas en 2-exploration)
CENTINELAS_NULOS: list[str] = [
    "",
    "?",
    "-",
    "--",
    ".",
    "na",
    "n/a",
    "#n/a",
    "nan",
    "null",
    "none",
    "missing",
    "unknown",
    "desconocido",
    "sin dato",
    "ninguno",
]

CATEGORIAS_RATING: list[str] = ["1", "2", "3", "4", "5"]
COLS_ENTERAS: list[str] = ["gre_score", "toefl_score"]
COLS_FLOTANTES: list[str] = ["sop", "lor", "cgpa", TARGET]
COLS_BOOLEANAS: list[str] = ["research"]
MAPA_BOOLEANO: dict[str, bool] = {"0": False, "1": True}

# dominio documentado en data/01_raw/Informacion.txt, incluido el objetivo
RANGOS: dict[str, tuple[float, float]] = {
    "gre_score": (0, 340),
    "toefl_score": (0, 120),
    "sop": (1, 5),
    "lor": (1, 5),
    "cgpa": (0, 10),
    TARGET: (0, 1),
}

# atributos derivados que anade `agregar_atributos_derivados` (ver 4-feat_eng)
COLUMNAS_DERIVADAS: list[str] = ["indice_academico", "sop_lor_media", "rating_x_research"]

COLS_PREDICTORAS: list[str] = [columna for columna in COLUMNAS_ESPERADAS if columna != TARGET]

# categorias validas de university_rating, ya como numero (el tipado las guarda como texto)
CATEGORIAS_VALIDAS_RATING: tuple[float, ...] = tuple(
    float(categoria) for categoria in CATEGORIAS_RATING
)

# minimo de predictores conocidos para que un registro sea utilizable: 4 de 7. El peor
# registro de la fuente actual tiene exactamente 4, asi que el umbral no descarta nada hoy
# y detecta manana una extraccion que llegue medio vacia.
MIN_PREDICTORES_CONOCIDOS = 4

# margen para comparar atributos derivados con su formula, por el ruido del punto flotante
TOLERANCIA_DERIVADOS = 1e-9

RUTA_ENTRADA_RELATIVA = Path("data") / "01_raw" / "Admission_Predict.csv"
RUTA_SALIDA_RELATIVA = Path("data") / "04_feature" / "admisiones_features.parquet"


def buscar_raiz_proyecto() -> Path:
    """Sube por el árbol de directorios hasta encontrar la raíz del repositorio."""
    actual = Path(__file__).resolve()
    for candidato in actual.parents:
        if (candidato / "pyproject.toml").exists():
            return candidato
    raise FileNotFoundError("No se encontro la raiz del proyecto")


def leer_datos_crudos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV como texto, sin conversión automática de tipos ni de nulos.

    Leer todo como `object` con `na_filter=False` evita que pandas decida por su cuenta
    qué es un nulo y qué tipo tiene cada columna: esas dos decisiones se toman explícitamente
    más abajo, que es la única forma de auditarlas.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos crudos en {ruta}")
    datos = pd.read_csv(ruta, dtype="object", na_filter=False)
    logger.info("Leidas %d filas x %d columnas de %s", len(datos), datos.shape[1], ruta)
    return datos


def a_snake_case(nombre: str) -> str:
    """Normaliza un nombre de columna: sin espacios sobrantes y en minúsculas.

    >>> a_snake_case("LOR ")
    'lor'
    """
    return nombre.strip().lower().replace(" ", "_")


def normalizar_nombres(datos: pd.DataFrame) -> pd.DataFrame:
    """Renombra las columnas a snake_case y verifica que estén todas las esperadas.

    Falla si falta alguna columna: es preferible romper aquí, con un mensaje claro, que
    generar un parquet incompleto que reviente tres etapas más adelante.
    """
    renombrado = datos.rename(columns=a_snake_case)
    faltantes = [columna for columna in COLUMNAS_ESPERADAS if columna not in renombrado.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas en la fuente: {faltantes}")
    return renombrado[COLUMNAS_ESPERADAS]


def unificar_nulos(datos: pd.DataFrame) -> pd.DataFrame:
    """Convierte a `pd.NA` cualquier representación textual de dato ausente.

    Un `'n/a'` y un `'-'` significan lo mismo que una celda vacía, pero sin unificarlos
    sobrevivirían como categorías propias y ensuciarían el tipado.
    """
    patron = {valor.strip().lower() for valor in CENTINELAS_NULOS}
    unificado = datos.copy()
    for columna in unificado.columns:
        serie = unificado[columna].astype("string").str.strip()
        unificado[columna] = serie.mask(serie.str.lower().isin(patron), pd.NA)
    logger.info("Celdas nulas tras unificar centinelas: %d", int(unificado.isna().sum().sum()))
    return unificado


def eliminar_duplicados(datos: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas idénticas en todas sus columnas.

    El dataset no tiene identificador de aspirante, así que dos filas iguales son
    indistinguibles: mantenerlas duplicaría el peso de ese perfil al entrenar.
    """
    sin_duplicados = datos.drop_duplicates().reset_index(drop=True)
    logger.info("Duplicados exactos eliminados: %d", len(datos) - len(sin_duplicados))
    return sin_duplicados


def tipar_columnas(datos: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada columna su tipo nullable de pandas.

    `university_rating` queda como categórica ordinal, `research` como booleana y el resto
    como `Int64`/`Float64`. Antes de convertir se guarda el número de nulos de cada columna
    para detectar los valores que `errors="coerce"` habría descartado en silencio.

    El tipado es también una validación de formato: un `'9'` en `university_rating`, un
    `'2'` en `research` o un `'trescientos'` en `gre_score` no se convierten en nulos sin
    más, se reportan como `ErrorValidacion` con el detalle de los valores culpables.

    Las columnas que no estén presentes se omiten, de modo que la misma función sirve para
    los datos de entrenamiento y para los de inferencia, a los que les falta el objetivo.
    """
    nulos_antes = datos.isna().sum()
    tipado = datos.copy()
    violaciones: list[Violacion] = []

    # las categorias se comprueban ANTES de construir la categorica: al construirla, un
    # valor no declarado se convertiria en nulo y el error quedaria enterrado
    vistas = (
        set(tipado["university_rating"].dropna().astype("string"))
        if "university_rating" in tipado.columns
        else set()
    )
    intrusas = sorted(vistas - set(CATEGORIAS_RATING))
    if intrusas:
        violaciones.append(
            Violacion(
                "categoria_invalida",
                "university_rating",
                f"valores no declarados {intrusas}, permitidos {CATEGORIAS_RATING}",
            )
        )
    elif "university_rating" in tipado.columns:
        tipado["university_rating"] = pd.Categorical(
            tipado["university_rating"], categories=CATEGORIAS_RATING, ordered=True
        )

    for columna in [c for c in COLS_ENTERAS if c in tipado.columns]:
        tipado[columna] = pd.to_numeric(tipado[columna], errors="coerce").astype("Int64")
    for columna in [c for c in COLS_FLOTANTES if c in tipado.columns]:
        tipado[columna] = pd.to_numeric(tipado[columna], errors="coerce").astype("Float64")
    for columna in [c for c in COLS_BOOLEANAS if c in tipado.columns]:
        inesperados = sorted(set(tipado[columna].dropna().unique()) - set(MAPA_BOOLEANO))
        if inesperados:
            violaciones.append(
                Violacion(
                    "valor_no_booleano",
                    columna,
                    f"valores fuera de {sorted(MAPA_BOOLEANO)}: {inesperados}",
                )
            )
        else:
            tipado[columna] = tipado[columna].map(MAPA_BOOLEANO).astype("boolean")

    diferencia = tipado.isna().sum() - nulos_antes
    for columna, perdidos in diferencia[diferencia > 0].items():
        malos = datos.loc[datos[columna].notna() & tipado[columna].isna(), columna]
        violaciones.append(
            Violacion(
                "valor_no_convertible",
                str(columna),
                f"texto que no se pudo convertir: {malos.unique()[:5].tolist()}",
                int(perdidos),
            )
        )
    if violaciones:
        raise ErrorValidacion(violaciones, contexto="el tipado de la fuente")
    return tipado


def _al_menos_predictores(datos: pd.DataFrame, columnas: list[str], minimo: int) -> pd.Series:
    """Serie booleana: `True` en las filas con al menos `minimo` de esas columnas conocidas."""
    return datos[columnas].notna().sum(axis=1) >= minimo


def _coincide(calculado: pd.Series, guardado: pd.Series) -> pd.Series:
    """Compara dos columnas numéricas tolerando el ruido del punto flotante.

    Las filas en las que alguno de los dos valores es nulo se dan por válidas: un derivado
    nulo porque su entrada lo era no es una incoherencia, es la propagación esperada.
    """
    diferencia = (calculado - guardado).abs()
    return (diferencia <= TOLERANCIA_DERIVADOS) | calculado.isna() | guardado.isna()


# --- Reglas de integridad entre campos (se evaluan solo si estan sus columnas) ----------

RELACION_REGISTRO_UTILIZABLE = ReglaRelacion(
    nombre="registro_utilizable",
    descripcion=(
        f"cada fila debe tener al menos {MIN_PREDICTORES_CONOCIDOS} de los "
        f"{len(COLS_PREDICTORAS)} predictores conocidos"
    ),
    columnas=tuple(COLS_PREDICTORAS),
    condicion=lambda datos: _al_menos_predictores(
        datos, COLS_PREDICTORAS, MIN_PREDICTORES_CONOCIDOS
    ),
)

RELACION_PERFIL_ACADEMICO = ReglaRelacion(
    nombre="perfil_academico_minimo",
    descripcion="sin gre_score, toefl_score ni cgpa no hay nada que estimar",
    columnas=("gre_score", "toefl_score", "cgpa"),
    condicion=lambda datos: _al_menos_predictores(datos, ["gre_score", "toefl_score", "cgpa"], 1),
)

RELACION_SOP_LOR_MEDIA = ReglaRelacion(
    nombre="sop_lor_media_coherente",
    descripcion="sop_lor_media debe ser el promedio exacto de sop y lor",
    columnas=("sop", "lor", "sop_lor_media"),
    condicion=lambda datos: _coincide((datos["sop"] + datos["lor"]) / 2, datos["sop_lor_media"]),
)

RELACION_RATING_RESEARCH = ReglaRelacion(
    nombre="rating_x_research_coherente",
    descripcion="rating_x_research debe ser el producto de university_rating y research",
    columnas=("university_rating", "research", "rating_x_research"),
    condicion=lambda datos: _coincide(
        datos["university_rating"] * datos["research"], datos["rating_x_research"]
    ),
)

RELACION_INDICE_ACADEMICO = ReglaRelacion(
    nombre="indice_academico_coherente",
    descripcion="indice_academico debe seguir la formula documentada en 4-feat_eng",
    columnas=("cgpa", "gre_score", "toefl_score", "indice_academico"),
    condicion=lambda datos: _coincide(
        (datos["cgpa"] / 10 + (datos["gre_score"] - 260) / (340 - 260) + datos["toefl_score"] / 120)
        / 3,
        datos["indice_academico"],
    ),
)

# --- Contrato de la fuente, tras tipar (dtypes nullable de pandas) ----------------------

ESQUEMA_FUENTE = EsquemaDatos(
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
        TARGET: ReglaColumna(tipos=("Float64",), rango=RANGOS[TARGET], max_nulos=0.0),
    },
    filas_unicas=True,
    relaciones=(RELACION_REGISTRO_UTILIZABLE, RELACION_PERFIL_ACADEMICO),
)

# --- Contrato de las features, lo que se persiste (float64 con np.nan) ------------------

ESQUEMA_FEATURES = EsquemaDatos(
    columnas={
        "gre_score": ReglaColumna(tipos=("float64",), rango=RANGOS["gre_score"]),
        "toefl_score": ReglaColumna(tipos=("float64",), rango=RANGOS["toefl_score"]),
        "university_rating": ReglaColumna(tipos=("float64",), categorias=CATEGORIAS_VALIDAS_RATING),
        "sop": ReglaColumna(tipos=("float64",), rango=RANGOS["sop"]),
        "lor": ReglaColumna(tipos=("float64",), rango=RANGOS["lor"]),
        "cgpa": ReglaColumna(tipos=("float64",), rango=RANGOS["cgpa"]),
        "research": ReglaColumna(tipos=("float64",), categorias=(0.0, 1.0)),
        "indice_academico": ReglaColumna(tipos=("float64",), obligatoria=False),
        "sop_lor_media": ReglaColumna(tipos=("float64",), rango=(1, 5), obligatoria=False),
        "rating_x_research": ReglaColumna(tipos=("float64",), rango=(0, 5), obligatoria=False),
        TARGET: ReglaColumna(tipos=("float64",), rango=RANGOS[TARGET], max_nulos=0.0),
    },
    filas_unicas=True,
    relaciones=(
        RELACION_REGISTRO_UTILIZABLE,
        RELACION_PERFIL_ACADEMICO,
        RELACION_SOP_LOR_MEDIA,
        RELACION_RATING_RESEARCH,
        RELACION_INDICE_ACADEMICO,
    ),
)


def validar_fuente(datos: pd.DataFrame) -> pd.DataFrame:
    """Primera puerta de calidad: la fuente, ya tipada, debe cumplir `ESQUEMA_FUENTE`.

    Comprueba tipos, dominio documentado en `data/01_raw/Informacion.txt`, categorías
    válidas, porcentaje de nulos, unicidad de registros e integridad entre campos.

    Es una verificación, no una corrección: si la fuente trae un GRE de 900 hay un problema
    de captura que merece mirarse, no taparse. El recorte defensivo se aplica después, en
    `construir_features`, pensando en los datos que lleguen en inferencia.
    """
    return validar(datos, ESQUEMA_FUENTE, contexto="la fuente tipada")


def validar_features(features: pd.DataFrame) -> pd.DataFrame:
    """Segunda puerta de calidad: lo que está a punto de persistirse.

    Se ejecuta **antes** de escribir el parquet, de modo que un fallo deja el archivo
    anterior intacto en vez de sustituirlo por features corruptas. Además de repetir las
    comprobaciones de dominio sobre el resultado, verifica que los atributos derivados
    siguen la fórmula documentada: si alguien cambia una de esas fórmulas y se olvida de la
    otra mitad, la incoherencia aparece aquí y no tres etapas más adelante.
    """
    return validar(features, ESQUEMA_FEATURES, contexto="las features a persistir")


def construir_features(datos: pd.DataFrame, *, con_derivados: bool = False) -> pd.DataFrame:
    """Convierte el dataset tipado en la matriz de features, con el objetivo al final.

    Reutiliza las funciones de `src/data/transformaciones.py` —las mismas que están dentro
    del pipeline serializado de `4-feat_eng`— para que las features guardadas y las que se
    calculan en inferencia salgan del mismo código.

    Parameters
    ----------
    con_derivados:
        Añade `indice_academico`, `sop_lor_media` y `rating_x_research`. Por defecto está
        desactivado: en `4-feat_eng` los tres atributos derivados no mejoraron el MAE de
        validación cruzada (0.0467 frente a 0.0468, dentro del ruido de ±0.005), y el
        artefacto entrenado espera exactamente los siete predictores originales.
    """
    predictores = a_tipos_sklearn(datos.drop(columns=[TARGET]))
    predictores = recortar_a_dominio(predictores)
    if con_derivados:
        predictores = agregar_atributos_derivados(predictores)
    features = predictores.copy()
    features[TARGET] = datos[TARGET].astype("float64").to_numpy()
    logger.info("Features construidas: %d filas x %d columnas", *features.shape)
    return features


def guardar_features(features: pd.DataFrame, ruta: Path) -> Path:
    """Escribe la matriz de features en parquet, creando el directorio si hace falta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(ruta, index=False, engine="pyarrow", compression="snappy")
    logger.info("Guardado %s (%.1f KB)", ruta, ruta.stat().st_size / 1024)
    return ruta


def ejecutar_pipeline(
    ruta_entrada: Path,
    ruta_salida: Path,
    *,
    con_derivados: bool = False,
) -> pd.DataFrame:
    """Ejecuta el pipeline completo y devuelve las features que acaba de guardar.

    El orden importa: las tres validaciones —la fuente, las features y la integridad entre
    ambas— ocurren **antes** de `guardar_features`, así que un dato inválido detiene el
    proceso sin escribir nada y el parquet anterior sigue siendo el último bueno conocido.
    """
    datos = leer_datos_crudos(ruta_entrada)
    datos = normalizar_nombres(datos)
    datos = unificar_nulos(datos)
    datos = eliminar_duplicados(datos)
    datos = tipar_columnas(datos)
    datos = validar_fuente(datos)
    features = construir_features(datos, con_derivados=con_derivados)
    features = validar_features(features)
    comparar_datasets(
        datos,
        features,
        columnas_conservadas=COLUMNAS_ESPERADAS,
        contexto="la transformacion de fuente a features",
    )
    guardar_features(features, ruta_salida)
    return features


def _parsear_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    """Define la interfaz de línea de comandos del script."""
    raiz = buscar_raiz_proyecto()
    parser = argparse.ArgumentParser(
        description="Feature pipeline: datos crudos de admisiones -> features en parquet."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=raiz / RUTA_ENTRADA_RELATIVA,
        help="CSV de datos crudos (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=raiz / RUTA_SALIDA_RELATIVA,
        help="Parquet de features a generar (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--con-derivados",
        action="store_true",
        help="Anade los atributos derivados de 4-feat_eng (indice_academico, etc.)",
    )
    return parser.parse_args(argumentos)


def main(argumentos: list[str] | None = None) -> int:
    """Punto de entrada del script. Devuelve 0 si el pipeline terminó bien, 1 si no.

    Un fallo de validación no es un error de programación, sino el resultado normal de un
    dato malo: se reporta como un informe legible y un código de salida distinto de cero
    —lo que necesita un orquestador para detener el flujo— en vez de como un *traceback*.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    opciones = _parsear_argumentos(argumentos)
    try:
        features = ejecutar_pipeline(
            opciones.entrada, opciones.salida, con_derivados=opciones.con_derivados
        )
    except ErrorValidacion as error:
        # el informe se registra fuera del `except` a proposito: un dato malo no es un fallo
        # del programa, asi que el traceback seria ruido y el informe es la informacion util
        informe = str(error)
    else:
        logger.info("Feature pipeline terminado: %s", list(features.columns))
        return 0
    logger.error("%s", informe)
    logger.error("No se persistieron features: %s no se ha modificado", opciones.salida)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
