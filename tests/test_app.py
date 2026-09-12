import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

TIEMPO_LIMITE = 180
PESTANAS = 2
ASPIRANTES_EJEMPLO = 10
TABLAS_DEL_LOTE = 2

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO_EJEMPLO = RAIZ / "examples" / "aspirantes_ejemplo.csv"
ARCHIVO_PREDICCIONES = RAIZ / "examples" / "predicciones_ejemplo.csv"

CABECERA = "id,GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research"
FILA_CON_GRE_IMPOSIBLE = "A-1,900,113,4,4.5,4.5,9.1,1"
FILA_SIN_PERFIL = "A-2,n/a,n/a,n/a,n/a,3,8.1,n/a"


def _abrir_demo() -> AppTest:
    """Ejecuta app.py en el entorno de pruebas de Streamlit."""
    prueba = AppTest.from_file("../app.py", default_timeout=TIEMPO_LIMITE)
    return prueba.run()


def test_la_demo_carga_sin_errores() -> None:
    """El formulario se renderiza y el modelo se carga al abrir la página."""
    demo = _abrir_demo()

    assert not demo.exception
    assert "admitido" in demo.title[0].value
    assert len(demo.button) >= 1


def test_al_enviar_el_formulario_se_obtiene_una_prediccion() -> None:
    """El caso de uso completo: enviar el formulario devuelve probabilidad y cesta."""
    demo = _abrir_demo()

    demo.button[0].click().run()

    assert not demo.exception
    assert len(demo.metric) == 1
    assert demo.metric[0].value.endswith("%")
    texto_completo = " ".join(elemento.value for elemento in demo.markdown)
    assert any(cesta in texto_completo for cesta in ("segura", "probable", "ambiciosa"))


def _subir(demo: AppTest, contenido: bytes, nombre: str = "aspirantes.csv") -> AppTest:
    """Simula que alguien sube un archivo en la pestaña de lotes."""
    demo.get("file_uploader")[0].set_value((nombre, contenido, "text/csv"))
    return demo.run()


def _csv(filas: str) -> bytes:
    """Archivo CSV en memoria con la cabecera real del dataset."""
    return f"{CABECERA}\n{filas}\n".encode()


def test_un_perfil_debil_muestra_la_advertencia() -> None:
    """Por debajo del umbral, la interfaz avisa de que el modelo es optimista."""
    demo = _abrir_demo()

    for control in demo.slider:
        control.set_value(control.min)
    demo.select_slider[0].set_value(1)
    demo.radio[0].set_value("No")
    demo.button[0].click().run()

    assert not demo.exception
    assert len(demo.warning) == 1
    assert "precaución" in demo.warning[0].value


# --- Estructura de dos pestanas ---------------------------------------------------------


def test_la_demo_tiene_las_dos_pestanas() -> None:
    """Una para el formulario y otra para el lote, como pide el enunciado."""
    demo = _abrir_demo()

    etiquetas = [pestana.label for pestana in demo.tabs]

    assert len(etiquetas) == PESTANAS
    assert "individual" in etiquetas[0]
    assert "lotes" in etiquetas[1].lower()


def test_la_pestana_de_lotes_ofrece_el_archivo_de_ejemplo() -> None:
    """Nadie debería tener que fabricarse un archivo para probar la demo."""
    demo = _abrir_demo()

    descargas = [boton.label for boton in demo.get("download_button")]

    assert any("ejemplo" in etiqueta for etiqueta in descargas)
    assert len(demo.get("file_uploader")) == 1


def test_el_selector_solo_admite_los_formatos_soportados() -> None:
    """El propio control impide subir un .xlsx, así que ese error no llega a ocurrir."""
    demo = _abrir_demo()

    assert demo.get("file_uploader")[0].allowed_type == [".csv", ".parquet"]


# --- Orquestador del lote: archivo valido -----------------------------------------------


def test_un_archivo_valido_produce_la_tabla_y_la_descarga() -> None:
    """El caso de uso completo de la pestaña: subir, ver resultados y poder descargarlos."""
    demo = _subir(_abrir_demo(), ARCHIVO_EJEMPLO.read_bytes(), ARCHIVO_EJEMPLO.name)

    assert not demo.exception
    metricas = {metrica.label: metrica.value for metrica in demo.metric}
    assert metricas["Aspirantes procesados"] == str(ASPIRANTES_EJEMPLO)
    assert metricas["Probabilidad media"].endswith("%")
    assert len(demo.dataframe) == TABLAS_DEL_LOTE  # resumen por cesta y resultados
    assert any("predicciones" in boton.label.lower() for boton in demo.get("download_button"))


def test_el_lote_devuelve_las_columnas_de_salida_del_pipeline() -> None:
    """Las cinco columnas que añade el modelo, con el contexto del archivo por delante."""
    demo = _subir(_abrir_demo(), ARCHIVO_EJEMPLO.read_bytes(), ARCHIVO_EJEMPLO.name)

    tabla = demo.dataframe[1].value

    assert len(tabla) == ASPIRANTES_EJEMPLO
    assert {"prediccion", "cesta", "limite_inferior", "limite_superior", "advertencia"} <= set(
        tabla.columns
    )
    assert tabla["id"].tolist()[0] == "A-001"  # el contexto vuelve a salir


def test_lo_que_se_descarga_coincide_con_el_ejemplo_versionado() -> None:
    """La evidencia del criterio de aceptación: mismo archivo dentro y fuera de la demo.

    El botón de descarga serializa esta misma tabla con `to_csv`, así que comparar el CSV
    resultante con `examples/predicciones_ejemplo.csv` comprueba lo que se descargaría.
    """
    demo = _subir(_abrir_demo(), ARCHIVO_EJEMPLO.read_bytes(), ARCHIVO_EJEMPLO.name)

    descargado = pd.read_csv(io.StringIO(demo.dataframe[1].value.to_csv(index=False)))

    pd.testing.assert_frame_equal(descargado, pd.read_csv(ARCHIVO_PREDICCIONES))


def test_el_resumen_por_cesta_reparte_todo_el_lote() -> None:
    """El resumen que se mira primero: cuántos caen en cada opción."""
    demo = _subir(_abrir_demo(), ARCHIVO_EJEMPLO.read_bytes(), ARCHIVO_EJEMPLO.name)

    resumen = demo.dataframe[0].value

    assert resumen["n"].sum() == ASPIRANTES_EJEMPLO
    assert set(resumen["cesta"]) <= {"segura", "probable", "ambiciosa"}


def test_el_lote_avisa_de_las_predicciones_del_tramo_bajo() -> None:
    """El archivo de ejemplo trae un aspirante flojo a propósito: la demo debe marcarlo."""
    demo = _subir(_abrir_demo(), ARCHIVO_EJEMPLO.read_bytes(), ARCHIVO_EJEMPLO.name)

    avisos = " ".join(aviso.value for aviso in demo.warning)

    assert "advertencia" in avisos


# --- Orquestador del lote: archivo invalido ---------------------------------------------


def test_un_valor_fuera_de_rango_muestra_un_error_legible_y_la_app_sigue_en_pie() -> None:
    """Es preferible rechazar el archivo entero que publicar una predicción inservible."""
    demo = _subir(_abrir_demo(), _csv(FILA_CON_GRE_IMPOSIBLE))

    assert not demo.exception
    assert demo.error, "el fallo tiene que verse en la interfaz"
    detalle = " ".join(elemento.value for elemento in demo.markdown)
    assert "fuera_de_rango" in detalle
    assert "gre_score" in detalle
    assert len(demo.tabs) == PESTANAS  # la aplicacion sigue dibujandose entera


def test_el_archivo_invalido_no_genera_tabla_ni_descarga() -> None:
    """Si no hay predicciones, no puede haber nada que descargar."""
    demo = _subir(_abrir_demo(), _csv(FILA_CON_GRE_IMPOSIBLE))

    assert not demo.dataframe
    assert not any("predicciones" in boton.label.lower() for boton in demo.get("download_button"))


def test_un_perfil_casi_vacio_tambien_se_rechaza() -> None:
    """Con menos de cuatro predictores conocidos no hay perfil que puntuar."""
    demo = _subir(_abrir_demo(), _csv(FILA_SIN_PERFIL))

    assert not demo.exception
    detalle = " ".join(elemento.value for elemento in demo.markdown)
    assert "registro_utilizable" in detalle


def test_un_archivo_sin_las_columnas_del_modelo_se_rechaza() -> None:
    """Un CSV de otra cosa no revienta la app: se explica qué columnas faltan."""
    demo = _subir(_abrir_demo(), b"columna_a,columna_b\n1,2\n")

    assert not demo.exception
    assert any("faltan columnas" in elemento.value for elemento in demo.error)


@pytest.mark.parametrize(
    "etiqueta", ["Aspirantes procesados", "Probabilidad media", "Con advertencia"]
)
def test_el_resumen_del_lote_muestra_las_tres_cifras(etiqueta: str) -> None:
    """Cuántos, cuánto y cuántos con aviso: lo que se mira antes de abrir la tabla."""
    demo = _subir(_abrir_demo(), ARCHIVO_EJEMPLO.read_bytes(), ARCHIVO_EJEMPLO.name)

    assert etiqueta in [metrica.label for metrica in demo.metric]
