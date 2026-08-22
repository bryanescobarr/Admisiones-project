from streamlit.testing.v1 import AppTest

TIEMPO_LIMITE = 120


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
