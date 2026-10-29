import os
from streamlit.testing.v1 import AppTest
from sesgo import demo
from tests.conftest import ROOT


def test_streamlit_pages_render_and_annotation_roundtrip(tmp_path, monkeypatch):
    dbp = tmp_path / "ui.db"
    demo.run_demo(dbp, out_dir=tmp_path, boot_B=30, verbose=False)
    # nueva muestra con tareas pendientes para probar el envio desde la interfaz
    from sesgo import annotation, db, sampling
    from sesgo.lexicon import load_lexicon
    c = db.connect(dbp)
    d = sampling.draw_sample(c, "ui", 99, 1, 0, load_lexicon(ROOT / "data/lexicon/lexicon_fixture.csv"), "Colombia",
                             exclude_draw_ids=(1, 2))
    annotation.assign_independent(c, d["draw_id"], ["anotador_1", "anotador_2"], k=2)
    c.close()
    monkeypatch.setenv("SESGO_DB", str(dbp))
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
    assert not at.exception and at.title[0].value == "Panel general"
    assert at.metric[0].value not in ("0", 0)
    at.sidebar.radio[0].set_value("Auditoría").run()
    assert not at.exception and at.title[0].value.startswith("Auditoría")
    at.sidebar.radio[0].set_value("Anotación").run()
    assert not at.exception
    at.button[0].click().run()  # Guardar y continuar con etiquetas por defecto (MARCADO sin tokens -> error controlado)
    assert not at.exception and len(at.error) == 1
