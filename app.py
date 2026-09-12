"""Demo del modelo de predicción de admisión a posgrado.

Ejecutar con:

    uv run streamlit run app.py
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

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

st.set_page_config(
    page_title="Predicción de admisión a posgrado", page_icon="🎓", layout="centered"
)

COLORES_CESTA = {"segura": "🟢", "probable": "🟡", "ambiciosa": "🔴"}
DESCRIPCION_CESTA = {
    "segura": "Tienes buenas posibilidades. Vale la pena incluirla como opción de respaldo.",
    "probable": "Es una opción realista, pero no garantizada. El grueso de tu lista debería estar aquí.",
    "ambiciosa": "Es una apuesta. Inclúyela si te ilusiona, pero no construyas tu lista sobre ella.",
}


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
                "sube tu probabilidad" if valor >= 0 else "la baja" for valor in aportes.to_numpy()
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

st.caption(
    f"Modelo: Extra Trees sobre 471 perfiles · MAE {MAE_MODELO:.4f} · "
    "generado por `train_pipeline.py` · "
    "[Código y notebooks](https://github.com/bryanescobarr/Admisiones-project)"
)
