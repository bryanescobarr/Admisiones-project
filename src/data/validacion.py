"""Motor de validación de datos: reglas declarativas y un informe completo de fallos.

Vive en `src/data/` —la capa de *data validation* de `src/README.md`— y no dentro del
feature pipeline, porque la pregunta "¿estos datos son utilizables?" se repite en cada
etapa: al leer la fuente, antes de persistir las features y, más adelante, antes de
entrenar o de puntuar en producción.

**Por qué un motor propio y no `pandera` o `great_expectations`.** Las tres opciones que
propone la [guía de validación de datos][guia] resuelven el mismo problema; el criterio
aquí fue no añadir una dependencia nueva para nueve reglas que caben en un módulo con
pruebas. El diseño es deliberadamente equivalente al declarativo de `pandera` —un esquema
de datos, no una cascada de `if`— así que migrar consiste en traducir `EsquemaDatos` a un
`DataFrameSchema`, sin tocar quien lo llama.

[guia]: https://joserzapata.github.io/courses/ciencia-datos-en-produccion/data-validation/

**Dos decisiones de comportamiento que importan:**

1. **Se recogen todas las violaciones antes de fallar.** Detenerse en la primera obliga a
   ejecutar el pipeline tantas veces como problemas tenga el archivo; el informe completo
   se arregla de una sola vez.
2. **Validar no es corregir.** Una regla que falla detiene el proceso y deja el mensaje;
   nunca imputa, recorta ni descarta filas por su cuenta. La corrección es una decisión
   del pipeline, y debe verse en el código del pipeline.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

# proporcion de nulos por encima de la cual una columna se considera inutilizable
MAX_NULOS_POR_DEFECTO = 0.10

# por debajo de este numero de filas, una proporcion de nulos no dice nada: en un conjunto
# de tres registros, una sola celda vacia ya es un 33 %. Los umbrales relativos se aplican
# solo a partir de aqui; los absolutos (`max_nulos = 0`) se exigen siempre.
MIN_FILAS_PARA_PROPORCION = 30


@dataclass(frozen=True)
class Violacion:
    """Una regla incumplida, con lo necesario para entender qué pasó sin abrir los datos."""

    regla: str
    columna: str
    detalle: str
    filas_afectadas: int = 0

    def __str__(self) -> str:
        ubicacion = f"[{self.columna}] " if self.columna else ""
        plural = "" if self.filas_afectadas == 1 else "s"
        filas = f" ({self.filas_afectadas} fila{plural})" if self.filas_afectadas else ""
        return f"{ubicacion}{self.regla}: {self.detalle}{filas}"


class ErrorValidacion(Exception):
    """Se lanza cuando un conjunto de datos incumple al menos una regla.

    El mensaje lleva el informe completo —una línea por violación— porque este error se lee
    casi siempre en la salida de un script o en el log de un job, no en un depurador.
    """

    def __init__(self, violaciones: Sequence[Violacion], contexto: str = "") -> None:
        self.violaciones = list(violaciones)
        self.contexto = contexto
        ubicacion = f" en {contexto}" if contexto else ""
        encabezado = (
            f"Validacion fallida{ubicacion}: {len(self.violaciones)} regla(s) incumplida(s)"
        )
        detalle = "\n".join(f"  - {violacion}" for violacion in self.violaciones)
        super().__init__(f"{encabezado}\n{detalle}")


@dataclass(frozen=True)
class ReglaColumna:
    """Contrato de una columna: qué tipo, qué valores y cuántos nulos se admiten.

    Parameters
    ----------
    tipos:
        dtypes aceptados, como texto (`"float64"`, `"Int64"`, `"category"`...). Vacío
        significa que el tipo no se comprueba.
    rango:
        `(minimo, maximo)` inclusivo admitido para las columnas numéricas.
    categorias:
        conjunto cerrado de valores permitidos, para las columnas categóricas.
    max_nulos:
        proporción máxima de nulos tolerada, entre 0 y 1.
    formato_fecha:
        patrón `strftime` que deben cumplir las fechas guardadas como texto.
    unica:
        `True` si la columna es una clave y no admite valores repetidos.
    obligatoria:
        `False` para columnas opcionales, que solo se validan si están presentes.
    """

    tipos: tuple[str, ...] = ()
    rango: tuple[float, float] | None = None
    categorias: tuple[float, ...] | None = None
    max_nulos: float = MAX_NULOS_POR_DEFECTO
    formato_fecha: str | None = None
    unica: bool = False
    obligatoria: bool = True


@dataclass(frozen=True)
class ReglaRelacion:
    """Regla de integridad entre campos de una misma fila.

    `condicion` recibe el DataFrame y devuelve una serie booleana con `True` en las filas
    que la cumplen. Solo se evalúa si están presentes todas las columnas que declara, de
    modo que las relaciones entre atributos opcionales no estorban cuando no se generan.
    """

    nombre: str
    descripcion: str
    columnas: tuple[str, ...]
    condicion: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class EsquemaDatos:
    """Contrato completo de un conjunto de datos.

    Parameters
    ----------
    columnas:
        regla por columna.
    permitir_columnas_extra:
        si `False`, una columna no declarada es un error: en un pipeline suele significar
        que la fuente cambió sin avisar.
    filas_unicas:
        prohíbe registros duplicados exactos.
    min_valores_por_fila:
        mínimo de campos no nulos que debe tener un registro para ser utilizable.
    relaciones:
        reglas de integridad entre campos.
    min_filas_para_proporcion:
        tamaño a partir del cual se exigen los umbrales relativos de nulos.
    """

    columnas: dict[str, ReglaColumna]
    permitir_columnas_extra: bool = False
    filas_unicas: bool = False
    min_valores_por_fila: int = 0
    relaciones: tuple[ReglaRelacion, ...] = field(default_factory=tuple)
    min_filas_para_proporcion: int = MIN_FILAS_PARA_PROPORCION


def _a_numerico(serie: pd.Series) -> pd.Series:
    """Serie comparable numéricamente, aunque venga como categórica o nullable."""
    return pd.to_numeric(serie, errors="coerce")


def validar_columnas(datos: pd.DataFrame, esquema: EsquemaDatos) -> list[Violacion]:
    """Comprueba que estén las columnas obligatorias y ninguna inesperada."""
    violaciones = []
    for nombre, regla in esquema.columnas.items():
        if regla.obligatoria and nombre not in datos.columns:
            violaciones.append(Violacion("columna_faltante", nombre, "no esta en los datos"))
    if not esquema.permitir_columnas_extra:
        extra = [columna for columna in datos.columns if columna not in esquema.columnas]
        if extra:
            violaciones.append(
                Violacion("columna_inesperada", "", f"columnas no declaradas: {extra}")
            )
    if not datos.columns.is_unique:
        repetidas = datos.columns[datos.columns.duplicated()].tolist()
        violaciones.append(Violacion("columna_duplicada", "", f"nombres repetidos: {repetidas}"))
    return violaciones


def validar_tipo(serie: pd.Series, nombre: str, regla: ReglaColumna) -> list[Violacion]:
    """El dtype de la columna debe ser uno de los declarados."""
    if not regla.tipos or str(serie.dtype) in regla.tipos:
        return []
    return [
        Violacion(
            "tipo_inesperado",
            nombre,
            f"dtype '{serie.dtype}', se esperaba uno de {list(regla.tipos)}",
        )
    ]


def validar_nulos(
    serie: pd.Series,
    nombre: str,
    regla: ReglaColumna,
    min_filas: int = MIN_FILAS_PARA_PROPORCION,
) -> list[Violacion]:
    """La proporción de nulos no puede superar el máximo declarado.

    Un `max_nulos` de 0 es una regla absoluta —esa columna no admite ni un solo nulo— y se
    exige siempre. Los umbrales relativos, en cambio, solo se aplican si el conjunto tiene
    al menos `min_filas`: en una muestra pequeña la proporción es puro ruido.
    """
    if len(serie) == 0:
        return []
    if regla.max_nulos > 0 and len(serie) < min_filas:
        return []
    proporcion = float(serie.isna().mean())
    if proporcion <= regla.max_nulos:
        return []
    return [
        Violacion(
            "exceso_de_nulos",
            nombre,
            f"{proporcion:.1%} de nulos, maximo admitido {regla.max_nulos:.1%}",
            int(serie.isna().sum()),
        )
    ]


def validar_rango(serie: pd.Series, nombre: str, regla: ReglaColumna) -> list[Violacion]:
    """Todos los valores conocidos deben caer dentro del rango documentado."""
    if regla.rango is None:
        return []
    minimo, maximo = regla.rango
    valores = _a_numerico(serie).dropna()
    fuera = valores[(valores < minimo) | (valores > maximo)]
    if fuera.empty:
        return []
    return [
        Violacion(
            "fuera_de_rango",
            nombre,
            f"valores fuera de [{minimo}, {maximo}]: {fuera.unique()[:5].tolist()}",
            len(fuera),
        )
    ]


def validar_categorias(serie: pd.Series, nombre: str, regla: ReglaColumna) -> list[Violacion]:
    """Los valores deben pertenecer al conjunto cerrado declarado."""
    if regla.categorias is None:
        return []
    permitidas = set(regla.categorias)
    valores = _a_numerico(serie).dropna()
    intrusas = sorted(set(valores.unique()) - permitidas)
    if not intrusas:
        return []
    return [
        Violacion(
            "categoria_invalida",
            nombre,
            f"valores no declarados {intrusas}, permitidos {sorted(permitidas)}",
            int(valores.isin(intrusas).sum()),
        )
    ]


def validar_formato_fecha(serie: pd.Series, nombre: str, regla: ReglaColumna) -> list[Violacion]:
    """Las fechas guardadas como texto deben cumplir el patrón declarado.

    El dataset de admisiones no tiene columnas temporales, así que hoy ninguna regla la
    activa. Se implementa igualmente porque la fuente es una exportación manual: el día que
    llegue una columna de fecha de postulación, el contrato ya sabe comprobarla.
    """
    if regla.formato_fecha is None:
        return []
    valores = serie.dropna().astype("string")
    convertidas = pd.to_datetime(valores, format=regla.formato_fecha, errors="coerce")
    invalidas = valores[convertidas.isna()]
    if invalidas.empty:
        return []
    return [
        Violacion(
            "formato_de_fecha_invalido",
            nombre,
            f"no cumplen '{regla.formato_fecha}': {invalidas.unique()[:5].tolist()}",
            len(invalidas),
        )
    ]


def validar_unicidad(serie: pd.Series, nombre: str, regla: ReglaColumna) -> list[Violacion]:
    """Una columna clave no admite valores repetidos."""
    if not regla.unica:
        return []
    repetidos = serie.dropna()
    duplicados = repetidos[repetidos.duplicated()]
    if duplicados.empty:
        return []
    return [
        Violacion(
            "clave_duplicada",
            nombre,
            f"valores repetidos: {duplicados.unique()[:5].tolist()}",
            len(duplicados),
        )
    ]


def validar_filas(datos: pd.DataFrame, esquema: EsquemaDatos) -> list[Violacion]:
    """Integridad a nivel de registro: filas duplicadas y filas sin información suficiente."""
    violaciones = []
    if esquema.filas_unicas:
        duplicadas = int(datos.duplicated().sum())
        if duplicadas:
            violaciones.append(
                Violacion("fila_duplicada", "", "hay registros identicos", duplicadas)
            )
    if esquema.min_valores_por_fila > 0:
        informativas = datos.notna().sum(axis=1)
        pobres = informativas[informativas < esquema.min_valores_por_fila]
        if not pobres.empty:
            violaciones.append(
                Violacion(
                    "fila_sin_informacion",
                    "",
                    f"filas con menos de {esquema.min_valores_por_fila} campos conocidos: "
                    f"indices {pobres.index[:5].tolist()}",
                    len(pobres),
                )
            )
    return violaciones


def validar_relaciones(datos: pd.DataFrame, esquema: EsquemaDatos) -> list[Violacion]:
    """Integridad entre campos: cada relación se evalúa si están sus columnas."""
    violaciones = []
    for relacion in esquema.relaciones:
        if not all(columna in datos.columns for columna in relacion.columnas):
            continue
        incumplen = ~relacion.condicion(datos).fillna(False).astype(bool)
        if incumplen.any():
            violaciones.append(
                Violacion(
                    "relacion_incumplida",
                    ", ".join(relacion.columnas),
                    f"{relacion.nombre}: {relacion.descripcion}",
                    int(incumplen.sum()),
                )
            )
    return violaciones


def revisar(datos: pd.DataFrame, esquema: EsquemaDatos) -> list[Violacion]:
    """Aplica el esquema completo y devuelve todas las violaciones encontradas.

    No lanza excepciones: sirve para inspeccionar o para construir informes. Quien quiera
    que el proceso se detenga usa `validar`.
    """
    violaciones = validar_columnas(datos, esquema)
    for nombre, regla in esquema.columnas.items():
        if nombre not in datos.columns:
            continue
        serie = datos[nombre]
        violaciones.extend(validar_tipo(serie, nombre, regla))
        violaciones.extend(
            validar_nulos(serie, nombre, regla, esquema.min_filas_para_proporcion)
        )
        violaciones.extend(validar_rango(serie, nombre, regla))
        violaciones.extend(validar_categorias(serie, nombre, regla))
        violaciones.extend(validar_formato_fecha(serie, nombre, regla))
        violaciones.extend(validar_unicidad(serie, nombre, regla))
    violaciones.extend(validar_filas(datos, esquema))
    violaciones.extend(validar_relaciones(datos, esquema))
    return violaciones


def validar(datos: pd.DataFrame, esquema: EsquemaDatos, contexto: str = "") -> pd.DataFrame:
    """Valida y devuelve los datos intactos, o lanza `ErrorValidacion` con el informe.

    Devolver el mismo DataFrame permite encadenar la validación dentro del pipeline sin
    variables intermedias, dejando explícito en el código dónde está cada puerta de calidad.
    """
    violaciones = revisar(datos, esquema)
    if violaciones:
        raise ErrorValidacion(violaciones, contexto)
    return datos


def comparar_datasets(
    origen: pd.DataFrame,
    destino: pd.DataFrame,
    *,
    columnas_conservadas: Sequence[str],
    contexto: str = "",
) -> pd.DataFrame:
    """Integridad entre datasets: que la transformación no invente ni pierda información.

    Comprueba tres cosas sobre el par (entrada, salida) de una etapa del pipeline:

    1. **No aparecen filas de la nada**: la salida no puede tener más registros que la
       entrada, porque aquí solo se limpia y se deriva, nunca se aumenta.
    2. **No se pierden columnas por el camino**: las columnas que la etapa debe conservar
       siguen presentes.
    3. **Los valores conservados son los mismos**: para cada columna conservada, el
       conjunto de valores de la salida está contenido en el de la entrada. Detecta el
       fallo silencioso más caro de un pipeline —una transformación que altera una columna
       que creía estar solo copiando— y también los nulos que aparecen sin motivo.
    """
    violaciones: list[Violacion] = []
    if len(destino) > len(origen):
        violaciones.append(
            Violacion(
                "filas_inventadas",
                "",
                f"la salida tiene {len(destino)} filas y la entrada {len(origen)}",
            )
        )
    for columna in columnas_conservadas:
        if columna not in destino.columns:
            violaciones.append(Violacion("columna_perdida", columna, "no llego a la salida"))
            continue
        if columna not in origen.columns:
            continue
        valores_origen = set(_a_numerico(origen[columna]).dropna().round(6).unique())
        valores_destino = set(_a_numerico(destino[columna]).dropna().round(6).unique())
        nuevos = sorted(valores_destino - valores_origen)
        if nuevos:
            violaciones.append(
                Violacion(
                    "valor_no_presente_en_el_origen",
                    columna,
                    f"la salida introduce valores que no estaban en la entrada: {nuevos[:5]}",
                    len(nuevos),
                )
            )
        nulos_nuevos = int(destino[columna].isna().sum() - origen[columna].isna().sum())
        if nulos_nuevos > 0:
            violaciones.append(
                Violacion(
                    "nulos_introducidos",
                    columna,
                    "la transformacion genero nulos que no estaban en la entrada",
                    nulos_nuevos,
                )
            )
    if violaciones:
        raise ErrorValidacion(violaciones, contexto)
    return destino
