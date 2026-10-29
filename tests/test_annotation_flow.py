import pytest
from datetime import datetime, timedelta, timezone
from sesgo import annotation, sampling
from sesgo.annotation import submit_annotation
from sesgo.lexicon import load_lexicon
from sesgo.spans import Span
from tests.conftest import ROOT


@pytest.fixture()
def setup(loaded):
    entries = load_lexicon(ROOT / "data/lexicon/lexicon_fixture.csv")
    d = sampling.draw_sample(loaded, "m", 1, 3, 2, entries, "Colombia")
    annotation.add_annotators(loaded, ["a", "b", "c"])
    gid = annotation.register_guideline(loaded, "v1", ROOT / "docs/guia_anotacion_v1.md")
    annotation.assign_independent(loaded, d["draw_id"], ["a", "b", "c"], k=2)
    return loaded, gid


def test_each_review_gets_two_distinct_annotators(setup):
    c, _ = setup
    rows = c.execute("SELECT review_id, COUNT(DISTINCT annotator_id) n FROM assignment WHERE round=1 GROUP BY review_id").fetchall()
    assert rows and all(r["n"] == 2 for r in rows)


def test_lock_is_per_annotator_and_expires(setup):
    c, _ = setup
    t0 = datetime.now(timezone.utc)
    ta = annotation.next_task(c, "a", at=t0)
    # el MISMO anotador recibe la misma tarea (bloqueada), no otra
    assert annotation.next_task(c, "a", at=t0)["assignment_id"] == ta["assignment_id"]
    # otro anotador puede trabajar la misma resenia (segunda anotacion independiente)
    tb = [annotation.next_task(c, "b", at=t0) for _ in range(1)][0]
    assert tb is not None
    # el bloqueo vence y la tarea vuelve a la cola
    later = t0 + timedelta(minutes=30)
    annotation._expire_locks(c, later)
    assert c.execute("SELECT status FROM assignment WHERE id=?", (ta["assignment_id"],)).fetchone()[0] == "pending"


def test_submit_rules_and_independence(setup):
    c, gid = setup
    t = annotation.next_task(c, "a")
    with pytest.raises(ValueError):  # MARCADO sin tramo
        submit_annotation(c, t["assignment_id"], "a", gid, "MARCADO", "AUSENTE", [])
    with pytest.raises(PermissionError):  # tarea ajena
        submit_annotation(c, t["assignment_id"], "b", gid, "NO_MARCADO", "AUSENTE", [])
    submit_annotation(c, t["assignment_id"], "a", gid, "NO_MARCADO", "AUSENTE", [])
    with pytest.raises(ValueError):  # ya completada
        submit_annotation(c, t["assignment_id"], "a", gid, "NO_MARCADO", "AUSENTE", [])
    # independencia: b no puede leer la anotacion de a
    assert annotation.own_annotation(c, "b", t["review_id"]) is None
    assert annotation.own_annotation(c, "a", t["review_id"]) is not None


def test_disagreement_creates_third_round_task_and_gold(setup):
    c, gid = setup
    rid = c.execute("SELECT review_id FROM assignment WHERE round=1 GROUP BY review_id HAVING COUNT(*)=2 LIMIT 1").fetchone()[0]
    text = c.execute("SELECT text_norm FROM review WHERE id=?", (rid,)).fetchone()[0]
    who = [r[0] for r in c.execute("SELECT a.alias FROM assignment s JOIN annotator a ON a.id=s.annotator_id WHERE s.review_id=? AND s.round=1", (rid,))]
    ids = {r["alias"]: r["id"] for r in c.execute("SELECT s.id, a.alias FROM assignment s JOIN annotator a ON a.id=s.annotator_id WHERE s.review_id=? AND s.round=1", (rid,))}
    submit_annotation(c, ids[who[0]], who[0], gid, "NO_MARCADO", "AUSENTE", [])
    submit_annotation(c, ids[who[1]], who[1], gid, "MARCADO", "AUSENTE", [Span(0, 3, "DIALECT")])
    assert annotation.flag_disagreements(c) == 1
    third = c.execute("SELECT a.alias FROM assignment s JOIN annotator a ON a.id=s.annotator_id WHERE s.review_id=? AND s.round=2", (rid,)).fetchone()[0]
    assert third not in who
    tid = c.execute("SELECT s.id FROM assignment s JOIN annotator a ON a.id=s.annotator_id WHERE s.review_id=? AND s.round=2", (rid,)).fetchone()[0]
    submit_annotation(c, tid, third, gid, "MARCADO", "AUSENTE", [Span(0, 3, "DIALECT")])
    annotation.compute_gold(c)
    g = c.execute("SELECT dialect_gold, method FROM gold_label WHERE review_id=?", (rid,)).fetchone()
    assert g["dialect_gold"] == "MARCADO" and g["method"] == "mayoria"
