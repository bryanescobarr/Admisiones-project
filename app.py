"""Demo del modelo de predicción de admisión a posgrado.

Dos pestañas sobre el mismo modelo: **predicción individual**, para un aspirante que
rellena el formulario, y **procesamiento por lotes**, para quien orienta a varios a la vez y
sube un archivo.

**Aquí no se decide nada sobre el modelo.** Validar, predecir y clasificar en cestas ocurre
en `src/`: la pestaña individual llama a `src/inference/prediccion.py` y la de lotes encadena
las funciones de `src/pipelines/inference_pipeline/inference_pipeline.py`, las mismas que
ejecuta el script de línea de comandos. Si esta interfaz repitiera esa lógica, habría dos
fuentes de verdad y tarde o temprano darían números distintos para el mismo aspirante.

Ejecutar con:

    uv run streamlit run app.py
"""

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from data.validacion import ErrorValidacion  # noqa: E402
from inference.prediccion import (  # noqa: E402
    ETIQUETAS_CESTA,
    MAE_MODELO,
    NOMBRES_LEGIBLES,
    PREDICCION_MINIMA,
    UMBRAL_ADVERTENCIA,
    cargar_modelo,
    clasificar_cesta,
    contribuciones,
    intervalo_estimado,
    predecir,
    prediccion_media,
)
from pipelines.inference_pipeline.inference_pipeline import (  # noqa: E402
    COLUMNAS_SALIDA,
    EXTENSIONES_TABULARES,
    leer_datos_nuevos,
    predecir_lote,
    preparar_datos_nuevos,
    resumir_lote,
    separar_contexto,
)

st.set_page_config(
    page_title="Predicción de admisión a posgrado", page_icon="🎓", layout="centered"
)

# el texto de cada opción se indexa por las etiquetas que declara `src/inference/prediccion.py`,
# no por cadenas escritas aquí: los cortes y sus nombres son del modelo, no de la interfaz
COLORES_CESTA = dict(zip(ETIQUETAS_CESTA, ("🔴", "🟡", "🟢"), strict=True))
DESCRIPCION_CESTA = dict(
    zip(
        ETIQUETAS_CESTA,
        (
            "Es una apuesta. Inclúyela si te ilusiona, pero no construyas tu lista sobre ella.",
            "Es una opción realista, pero no garantizada. El grueso de tu lista debería estar aquí.",
            "Tienes buenas posibilidades. Vale la pena incluirla como opción de respaldo.",
        ),
        strict=True,
    )
)

# columnas que debe traer un archivo de lote, con su dominio documentado en Informacion.txt
COLUMNAS_ESPERADAS_LOTE = [
    ("GRE Score", "260 a 340"),
    ("TOEFL Score", "0 a 120"),
    ("University Rating", "1 a 5"),
    ("SOP", "1 a 5"),
    ("LOR", "1 a 5"),
    ("CGPA", "0 a 10"),
    ("Research", "0 o 1"),
]

ARCHIVO_EJEMPLO = RAIZ / "examples" / "aspirantes_ejemplo.csv"


@st.cache_resource
def obtener_modelo() -> tuple[Pipeline, float]:
    """Carga el modelo una sola vez y lo reutiliza entre ejecuciones."""
    modelo = cargar_modelo()
    return modelo, prediccion_media(modelo)


def campo_con_desconocido(
    etiqueta: str, ayuda: str, widget: Callable[..., Any], clave: str
) -> float | None:
    """Renderiza un control junto a una casilla 'No lo sé' que devuelve None."""
    columna_valor, columna_casilla = st.columns([3, 1])
    with columna_casilla:
        desconocido = st.checkbox("No lo sé", key=f"desconocido_{clave}")
    with columna_valor:
        valor = widget(etiqueta, disabled=desconocido, help=ayuda)
    return None if desconocido else valor


def procesar_lote(archivo_subido: Any, modelo: Pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encadena el inference pipeline sobre el archivo que subió la persona.

    `leer_datos_nuevos()` recibe una ruta, no bytes, así que el archivo se vuelca a un
    directorio temporal conservando su nombre —la extensión decide si se lee como CSV o como
    parquet— y se borra al salir. Nada se escribe en `data/`: el disco de Streamlit Cloud es
    efímero y la salida se entrega por descarga.

    Devuelve la tabla completa —contexto, predictores y predicciones— y el resumen por cesta.
    Cada paso es una función del pipeline: esta interfaz no valida, no predice y no clasifica.
    """
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = Path(carpeta) / archivo_subido.name
        ruta.write_bytes(archivo_subido.getvalue())
        datos = leer_datos_nuevos(ruta)

    predictores, contexto = separar_contexto(datos)
    predictores = preparar_datos_nuevos(predictores)
    predicciones = predecir_lote(modelo, predictores)
    salida = pd.concat([contexto, predictores, predicciones], axis=1)
    return salida, resumir_lote(salida)


def mostrar_violaciones(error: ErrorValidacion) -> None:
    """Muestra un fallo de validación como lo que es: una lista de reglas incumplidas.

    Se detallan una a una en vez de volcar el traceback, porque quien sube el archivo puede
    corregir «el GRE de la fila 3 está fuera de rango» y no puede hacer nada con una traza de
    Python.
    """
    st.error(
        f"**No se pudo procesar el archivo.** {len(error.violaciones)} regla(s) incumplida(s):"
    )
    st.markdown("\n".join(f"- {violacion}" for violacion in error.violaciones))
    st.caption(
        "El archivo se rechaza entero a propósito: es preferible corregir el dato a publicar "
        "una predicción que nadie debería usar."
    )


st.title("🎓 ¿Qué posibilidades tengo de ser admitido?")
st.markdown(
    "Estima tu probabilidad de admisión a un programa de posgrado a partir de tu perfil "
    "académico, y te dice si esa universidad es una opción **segura**, **probable** o "
    "**ambiciosa** para tu lista de postulaciones."
)

try:
    modelo, valor_base = obtener_modelo()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

pestana_individual, pestana_lotes = st.tabs(
    ["🎯 Predicción individual", "📂 Procesamiento por lotes"]
)

with pestana_individual:
    with st.form("perfil"):
        st.subheader("Tu perfil académico")
        st.caption(
            "Si te falta algún dato, marca «No lo sé»: el modelo trabaja igual con la información que tengas."
        )

        gre = campo_con_desconocido(
            "Puntaje GRE",
            "Graduate Record Examination, entre 260 y 340.",
            lambda etiqueta, disabled, help: st.slider(
                etiqueta, 260, 340, 316, disabled=disabled, help=help
            ),
            "gre",
        )
        toefl = campo_con_desconocido(
            "Puntaje TOEFL",
            "Examen de inglés, entre 0 y 120.",
            lambda etiqueta, disabled, help: st.slider(
                etiqueta, 0, 120, 107, disabled=disabled, help=help
            ),
            "toefl",
        )
        cgpa = campo_con_desconocido(
            "Promedio acumulado (CGPA)",
            "Promedio de tu pregrado sobre 10.",
            lambda etiqueta, disabled, help: st.slider(
                etiqueta, 0.0, 10.0, 8.6, 0.01, disabled=disabled, help=help
            ),
            "cgpa",
        )

        columna_izquierda, columna_derecha = st.columns(2)
        with columna_izquierda:
            rating = st.select_slider(
                "Calificación de tu universidad de origen",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="1 = menos reconocida, 5 = más reconocida.",
            )
            sop = st.slider("Fuerza de tu carta de intención (SOP)", 1.0, 5.0, 3.5, 0.5)
        with columna_derecha:
            research = st.radio(
                "¿Tienes experiencia en investigación?", ["Sí", "No"], index=0, horizontal=True
            )
            lor = st.slider("Fuerza de tus cartas de recomendación (LOR)", 1.0, 5.0, 3.5, 0.5)

        enviado = st.form_submit_button(
            "Calcular mi probabilidad", type="primary", use_container_width=True
        )

    if enviado:
        datos = {
            "gre_score": gre,
            "toefl_score": toefl,
            "university_rating": float(rating),
            "sop": sop,
            "lor": lor,
            "cgpa": cgpa,
            "research": 1.0 if research == "Sí" else 0.0,
        }
        probabilidad = predecir(modelo, datos)
        cesta = clasificar_cesta(probabilidad)
        limite_inferior, limite_superior = intervalo_estimado(probabilidad)

        st.divider()
        columna_numero, columna_cesta = st.columns([1, 2])
        with columna_numero:
            st.metric("Probabilidad estimada", f"{probabilidad:.0%}")
            st.caption(f"Rango de referencia: {limite_inferior:.0%} a {limite_superior:.0%}")
        with columna_cesta:
            st.markdown(f"### {COLORES_CESTA[cesta]} Opción {cesta}")
            st.write(DESCRIPCION_CESTA[cesta])

        if probabilidad < UMBRAL_ADVERTENCIA:
            st.warning(
                "**Toma este número con precaución.** En perfiles como el tuyo el modelo tiende a ser "
                f"optimista: casi no vio ejemplos por debajo de {PREDICCION_MINIMA:.0%} al entrenarse, y su "
                "error se duplica en este tramo. Tu probabilidad real podría ser menor."
            )

        st.subheader("¿Por qué este resultado?")
        st.caption(
            f"El modelo parte de {valor_base:.0%}, que es la probabilidad media de todos los aspirantes, "
            "y suma o resta según tu perfil."
        )

        aportes = contribuciones(modelo, datos)
        aportes.index = [NOMBRES_LEGIBLES.get(nombre, nombre) for nombre in aportes.index]

        figura, eje = plt.subplots(figsize=(7, 3.2))
        colores = ["tab:green" if valor >= 0 else "tab:red" for valor in aportes.to_numpy()[::-1]]
        eje.barh(aportes.index[::-1], aportes.to_numpy()[::-1], color=colores)
        eje.axvline(0, color="black", linewidth=0.8)
        eje.set_xlabel("aporte a tu probabilidad")
        figura.tight_layout()
        st.pyplot(figura)

        tabla = pd.DataFrame(
            {
                "Factor": aportes.index,
                "Aporte": [f"{valor:+.1%}" for valor in aportes.to_numpy()],
                "Efecto": [
                    "sube tu probabilidad" if valor >= 0 else "la baja"
                    for valor in aportes.to_numpy()
                ],
            }
        )
        st.dataframe(tabla, hide_index=True, use_container_width=True)

    with st.expander("Cómo leer este resultado (y sus límites)"):
        st.markdown(
            f"""
    **Qué es este número.** Una estimación basada en {len(ETIQUETAS_CESTA) and 471} perfiles de
    aspirantes reales. El modelo se equivoca en promedio **{MAE_MODELO:.1%}** en datos que nunca
    vio, así que trátalo como una orientación, no como un veredicto.

    **Para qué sirve de verdad.** Acierta la clasificación en segura / probable / ambiciosa el
    **85 %** de las veces, y nunca confundió una opción ambiciosa con una segura al evaluarlo.
    Es más fiable **ordenando** tus opciones que dando la cifra exacta.

    **Sus límites, dichos claramente:**

    - Los aspirantes de los datos tienen promedios altos (CGPA medio de 8.6/10). Si tu perfil
      queda muy por debajo, el modelo tiene poco en qué basarse.
    - No predice por debajo de **{PREDICCION_MINIMA:.0%}**: en el tramo bajo tiende a ser optimista.
    - No conoce tu carta de intención real, ni el programa concreto, ni el año de postulación.
      Dos aspirantes con datos idénticos pueden tener resultados distintos — y de hecho los
      tienen: en los datos originales difieren cerca de un 5 % entre sí.

    **No sustituye** el consejo de un asesor académico ni las estadísticas oficiales de cada
    universidad.
    """
        )

with pestana_lotes:
    st.subheader("Varios aspirantes de una sola vez")
    st.markdown(
        "Sube un archivo con **una fila por aspirante** y obtén todas las predicciones "
        "juntas, listas para descargar. Es la misma estimación de la otra pestaña aplicada "
        "en lote: **el mismo modelo y las mismas transformaciones**."
    )

    with st.expander("Qué debe tener el archivo", expanded=True):
        columnas = "\n".join(
            f"- `{nombre}` — {dominio}" for nombre, dominio in COLUMNAS_ESPERADAS_LOTE
        )
        st.markdown(
            f"""
Un archivo **{" o ".join(EXTENSIONES_TABULARES)}** con una fila por aspirante y estas columnas:

{columnas}

Detalles que suelen ahorrar un intento fallido:

- Los nombres valen tal cual salen del dataset original (`GRE Score`, `LOR `) o en minúsculas
  con guion bajo (`gre_score`, `lor`).
- **Puedes dejar celdas vacías** o escribir `n/a`: el modelo imputa el dato que falte, igual
  que cuando marcas «No lo sé» en la otra pestaña. Cada fila necesita al menos 4 de los 7 datos.
- Las **columnas de más se conservan**: si incluyes un `id` o el nombre del aspirante, volverán
  en el archivo de resultados para que sepas de quién es cada predicción.
"""
        )

    if ARCHIVO_EJEMPLO.exists():
        st.download_button(
            "⬇️ Descargar archivo de ejemplo",
            data=ARCHIVO_EJEMPLO.read_bytes(),
            file_name=ARCHIVO_EJEMPLO.name,
            mime="text/csv",
            help="Diez aspirantes de ejemplo, tres de ellos con datos incompletos.",
        )

    archivo = st.file_uploader(
        "Sube tu archivo", type=[extension.lstrip(".") for extension in EXTENSIONES_TABULARES]
    )

    if archivo is not None:
        try:
            predicciones_lote, resumen = procesar_lote(archivo, modelo)
        except ErrorValidacion as error:
            mostrar_violaciones(error)
        except (ValueError, FileNotFoundError) as error:
            st.error(f"**No se pudo procesar el archivo.**\n\n```\n{error}\n```")
        else:
            columna_total, columna_media, columna_aviso = st.columns(3)
            columna_total.metric("Aspirantes procesados", len(predicciones_lote))
            columna_media.metric(
                "Probabilidad media", f"{predicciones_lote['prediccion'].mean():.0%}"
            )
            columna_aviso.metric("Con advertencia", int(predicciones_lote["advertencia"].sum()))

            st.markdown("**Cómo se reparten las opciones**")
            st.dataframe(resumen, hide_index=True, width="stretch")
            st.bar_chart(resumen.set_index("cesta")["n"], height=200)

            st.markdown("**Resultados**")
            st.dataframe(
                predicciones_lote,
                hide_index=True,
                width="stretch",
                column_order=[
                    *[c for c in predicciones_lote.columns if c not in COLUMNAS_SALIDA],
                    *COLUMNAS_SALIDA,
                ],
            )

            st.download_button(
                "⬇️ Descargar predicciones (CSV)",
                data=predicciones_lote.to_csv(index=False).encode("utf-8"),
                file_name="predicciones.csv",
                mime="text/csv",
                type="primary",
            )

            if predicciones_lote["advertencia"].any():
                st.warning(
                    f"**{int(predicciones_lote['advertencia'].sum())} de "
                    f"{len(predicciones_lote)} predicciones caen por debajo de "
                    f"{UMBRAL_ADVERTENCIA:.0%}**, el tramo con menos ejemplos de "
                    "entrenamiento y más error. La columna `advertencia` las marca una a una."
                )

st.caption(
    f"Modelo: Extra Trees sobre 471 perfiles · MAE {MAE_MODELO:.4f} · "
    "generado por `train_pipeline.py` · "
    "[Código y notebooks](https://github.com/bryanescobarr/Admisiones-project)"
)
