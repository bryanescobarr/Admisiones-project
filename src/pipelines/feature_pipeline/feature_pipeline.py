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
    """
    nulos_antes = datos.isna().sum()
    tipado = datos.copy()

    tipado["university_rating"] = pd.Categorical(
        tipado["university_rating"], categories=CATEGORIAS_RATING, ordered=True
    )
    for columna in COLS_ENTERAS:
        tipado[columna] = pd.to_numeric(tipado[columna], errors="coerce").astype("Int64")
    for columna in COLS_FLOTANTES:
        tipado[columna] = pd.to_numeric(tipado[columna], errors="coerce").astype("Float64")
    for columna in COLS_BOOLEANAS:
        inesperados = set(tipado[columna].dropna().unique()) - set(MAPA_BOOLEANO)
        if inesperados:
            raise ValueError(f"Valores no booleanos en '{columna}': {sorted(inesperados)}")
        tipado[columna] = tipado[columna].map(MAPA_BOOLEANO).astype("boolean")

    diferencia = tipado.isna().sum() - nulos_antes
    perdidos = diferencia[diferencia > 0]
    if not perdidos.empty:
        raise ValueError(f"Valores descartados al convertir tipos: {perdidos.to_dict()}")
    return tipado


def validar_dominio(datos: pd.DataFrame) -> pd.DataFrame:
    """Comprueba que los datos crudos respetan el dominio documentado.

    Es una verificación, no una corrección: si la fuente trae un GRE de 900 hay un problema
    de captura que merece mirarse, no taparse. El recorte defensivo se aplica después, en
    `construir_features`, pensando en los datos que lleguen en inferencia.
    """
    fuera_de_rango = {}
    for columna, (minimo, maximo) in RANGOS.items():
        serie = datos[columna].dropna()
        invalidos = serie[(serie < minimo) | (serie > maximo)]
        if len(invalidos):
            fuera_de_rango[columna] = invalidos.tolist()[:5]
    if fuera_de_rango:
        raise ValueError(f"Valores fuera del dominio documentado: {fuera_de_rango}")

    categorias_vistas = set(datos["university_rating"].dropna().astype(str))
    if not categorias_vistas <= set(CATEGORIAS_RATING):
        raise ValueError(
            f"Categorias de university_rating fuera de lo declarado: "
            f"{sorted(categorias_vistas - set(CATEGORIAS_RATING))}"
        )
    if datos[TARGET].isna().any():
        raise ValueError(f"La variable objetivo '{TARGET}' no puede tener nulos")
    return datos


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
    """Ejecuta el pipeline completo y devuelve las features que acaba de guardar."""
    datos = leer_datos_crudos(ruta_entrada)
    datos = normalizar_nombres(datos)
    datos = unificar_nulos(datos)
    datos = eliminar_duplicados(datos)
    datos = tipar_columnas(datos)
    datos = validar_dominio(datos)
    features = construir_features(datos, con_derivados=con_derivados)
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
    """Punto de entrada del script. Devuelve 0 si el pipeline terminó bien."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    opciones = _parsear_argumentos(argumentos)
    features = ejecutar_pipeline(
        opciones.entrada, opciones.salida, con_derivados=opciones.con_derivados
    )
    logger.info("Feature pipeline terminado: %s", list(features.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
