from streamlit.testing.v1 import AppTest

from inference.lote import ARCHIVO_EJEMPLO_ENTRADA

TIEMPO_LIMITE = 120
ASPIRANTES_EJEMPLO = 8
PESTANAS = 2


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


# --- Pestana de procesamiento por lotes -------------------------------------------------


def _subir(demo: AppTest, contenido: bytes, nombre: str = "aspirantes.csv") -> AppTest:
    """Simula que alguien sube un archivo en la pestaña de lotes."""
    demo.get("file_uploader")[0].set_value((nombre, contenido, "text/csv"))
    return demo.run()


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


def test_al_subir_un_archivo_se_obtienen_las_predicciones_del_lote() -> None:
    """El caso de uso completo de la pestaña: subir, ver resultados y poder descargarlos."""
    demo = _subir(_abrir_demo(), ARCHIVO_EJEMPLO_ENTRADA.read_bytes(), "aspirantes_ejemplo.csv")

    assert not demo.exception
    metricas = {metrica.label: metrica.value for metrica in demo.metric}
    assert metricas["Aspirantes procesados"] == str(ASPIRANTES_EJEMPLO)
    assert metricas["Probabilidad media"].endswith("%")
    assert len(demo.dataframe) == 1
    assert any("predicciones" in boton.label.lower() for boton in demo.get("download_button"))


def test_el_lote_avisa_de_las_predicciones_del_tramo_bajo() -> None:
    """El archivo de ejemplo trae un aspirante flojo a proposito: la demo debe marcarlo."""
    demo = _subir(_abrir_demo(), ARCHIVO_EJEMPLO_ENTRADA.read_bytes(), "aspirantes_ejemplo.csv")

    avisos = " ".join(aviso.value for aviso in demo.warning)

    assert "tramo bajo" in avisos


def test_un_archivo_con_un_dato_imposible_no_genera_predicciones() -> None:
    """Es preferible rechazar el archivo entero que publicar una prediccion que nadie
    deberia usar."""
    malo = b"id,GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research\nA-1,900,113,4,4.5,4.5,9.1,1\n"

    demo = _subir(_abrir_demo(), malo)

    assert not demo.exception
    errores = " ".join(error.value for error in demo.error)
    assert "fuera_de_rango" in errores
    assert not demo.dataframe


def test_el_selector_solo_admite_los_formatos_soportados() -> None:
    """El propio control impide elegir un .xlsx, asi que el error nunca llega a ocurrir.

    La comprobacion de formato del modulo de lotes sigue existiendo como red de seguridad
    —se prueba en `tests/inference/test_lote.py`— pero aqui la interfaz lo evita antes.
    """
    demo = _abrir_demo()

    assert demo.get("file_uploader")[0].allowed_type == [".csv", ".parquet"]
