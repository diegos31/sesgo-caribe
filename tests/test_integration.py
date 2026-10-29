from sesgo import demo


def test_end_to_end_on_fixture(tmp_path):
    r = demo.run_demo(tmp_path / "e2e.db", out_dir=tmp_path, boot_B=60, verbose=False)
    assert r["ingesta"]["insertadas"] > 700
    assert r["particion"]["conteos"]["audit"] > 100
    assert sum(r["anotaciones_r1"].values()) > 400 and sum(r["anotaciones_r2"].values()) > 0
    a = r["auditorias"]["tfidf_lr"]
    assert a["n_marcado"] > 0 and a["n_no_marcado"] > 0
    assert a["decision_H1"].startswith("n_insuficiente")  # el fixture es demasiado pequeno para decidir
    assert (tmp_path / "informe_corrida.md").read_text(encoding="utf-8").count("SINTETICO") >= 1


def test_audit_set_never_leaks_into_training(tmp_path):
    from sesgo import db
    demo.run_demo(tmp_path / "e2e.db", out_dir=tmp_path, boot_B=30, verbose=False)
    c = db.connect(tmp_path / "e2e.db")
    n = c.execute("SELECT COUNT(*) FROM data_split a JOIN data_split b ON a.version=b.version "
                  "JOIN review ra ON ra.id=a.review_id JOIN review rb ON rb.id=b.review_id "
                  "WHERE a.role='audit' AND b.role IN ('train','dev','test') AND ra.dedupe_key=rb.dedupe_key").fetchone()[0]
    assert n == 0
    preds = c.execute("SELECT COUNT(*) FROM prediction p JOIN data_split s ON s.review_id=p.review_id WHERE s.role IN ('train','dev')").fetchone()[0]
    assert preds == 0  # solo se guardan predicciones de test y auditoria
