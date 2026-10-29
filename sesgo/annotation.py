"""Capa de servicio de anotacion: asignacion independiente, bloqueo por tarea,
envio validado, segunda/tercera anotacion y etiqueta final (gold).

Independencia: cada anotador solo puede leer SUS propias anotaciones hasta que
se calcula la etiqueta final. El bloqueo (lease) es por tarea y por anotador,
asi que nunca impide que otro anotador haga su anotacion independiente.
"""
from __future__ import annotations

import hashlib
import itertools
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .db import now
from .spans import Span, validate_spans

DIALECT_LABELS = ("MARCADO", "NO_MARCADO", "AMBIGUO")
PREJ_LABELS = ("PRESENTE", "AUSENTE", "AMBIGUO")
PREJ_TARGETS = ("NACIONALIDAD_PROCEDENCIA", "ETNIA_RAZA", "GENERO", "PROFESION_OFICIO",
                "NIVEL_SOCIOECONOMICO", "OTRO")


def add_annotators(conn: sqlite3.Connection, aliases: list[str]) -> None:
    for a in aliases:
        conn.execute("INSERT OR IGNORE INTO annotator(alias) VALUES (?)", (a,))
    conn.commit()


def register_guideline(conn: sqlite3.Connection, version: str, path: str | Path) -> int:
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    conn.execute("INSERT OR IGNORE INTO guideline_version(version, file_sha256, created_at) VALUES (?,?,?)",
                 (version, sha, now()))
    conn.commit()
    return conn.execute("SELECT id FROM guideline_version WHERE version=?", (version,)).fetchone()[0]


def _annotator_id(conn, alias: str) -> int:
    r = conn.execute("SELECT id FROM annotator WHERE alias=?", (alias,)).fetchone()
    if not r:
        raise ValueError(f"Anotador desconocido: {alias}")
    return r[0]


def assign_independent(conn: sqlite3.Connection, draw_id: int, aliases: list[str], k: int = 2,
                       seed: int = 0) -> int:
    """Cada resenia de la muestra se asigna a `k` anotadores distintos (ronda 1),
    rotando las combinaciones para equilibrar la carga."""
    ids = [_annotator_id(conn, a) for a in aliases]
    combos = list(itertools.combinations(ids, k))
    items = [r[0] for r in conn.execute("SELECT review_id FROM sample_item WHERE draw_id=? ORDER BY review_id",
                                        (draw_id,))]
    order = np.random.RandomState(seed).permutation(len(items))
    n = 0
    for pos, idx in enumerate(order):
        for aid in combos[pos % len(combos)]:
            c = conn.execute("INSERT OR IGNORE INTO assignment(review_id, annotator_id, round, status, created_at) "
                             "VALUES (?,?,1,'pending',?)", (items[idx], aid, now()))
            n += c.rowcount
    conn.commit()
    return n


def _expire_locks(conn, ts: datetime) -> None:
    conn.execute("UPDATE assignment SET status='pending', locked_until=NULL "
                 "WHERE status='locked' AND locked_until < ?", (ts.isoformat(timespec="seconds"),))


def next_task(conn: sqlite3.Connection, alias: str, lease_minutes: int = 20,
              at: datetime | None = None) -> dict | None:
    ts = at or datetime.now(timezone.utc)
    aid = _annotator_id(conn, alias)
    _expire_locks(conn, ts)
    row = conn.execute(
        "SELECT a.id, a.review_id, a.round, r.text_norm FROM assignment a JOIN review r ON r.id=a.review_id "
        "WHERE a.annotator_id=? AND a.status IN ('locked','pending') "
        "ORDER BY (a.status='locked') DESC, a.round DESC, a.id LIMIT 1", (aid,)).fetchone()
    if not row:
        conn.commit()
        return None
    until = (ts + timedelta(minutes=lease_minutes)).isoformat(timespec="seconds")
    conn.execute("UPDATE assignment SET status='locked', locked_until=? WHERE id=?", (until, row["id"]))
    conn.commit()
    return {"assignment_id": row["id"], "review_id": row["review_id"], "round": row["round"],
            "text": row["text_norm"], "locked_until": until}


def submit_annotation(conn: sqlite3.Connection, assignment_id: int, alias: str, guideline_id: int,
                      dialect_label: str, prejudice_label: str, spans: list[Span],
                      prejudice_target: str | None = None, comment: str | None = None) -> int:
    aid = _annotator_id(conn, alias)
    a = conn.execute("SELECT * FROM assignment WHERE id=?", (assignment_id,)).fetchone()
    if not a or a["annotator_id"] != aid:
        raise PermissionError("La tarea no pertenece a este anotador.")
    if a["status"] == "done":
        raise ValueError("La tarea ya fue completada.")
    if dialect_label not in DIALECT_LABELS or prejudice_label not in PREJ_LABELS:
        raise ValueError("Etiqueta invalida.")
    text = conn.execute("SELECT text_norm FROM review WHERE id=?", (a["review_id"],)).fetchone()[0]
    spans = validate_spans(text, spans)
    n_d = sum(s.layer == "DIALECT" for s in spans)
    n_p = sum(s.layer == "PREJ" for s in spans)
    if dialect_label == "MARCADO" and n_d == 0:
        raise ValueError("MARCADO exige al menos un tramo DIALECT.")
    if dialect_label == "NO_MARCADO" and n_d > 0:
        raise ValueError("NO_MARCADO no admite tramos DIALECT.")
    if prejudice_label == "PRESENTE" and (n_p == 0 or prejudice_target not in PREJ_TARGETS):
        raise ValueError("PRESENTE exige tramo PREJ y un objetivo valido.")
    if prejudice_label == "AUSENTE" and n_p > 0:
        raise ValueError("AUSENTE no admite tramos PREJ.")
    cur = conn.execute(
        "INSERT INTO annotation(assignment_id, review_id, annotator_id, guideline_id, dialect_label, "
        "prejudice_label, prejudice_target, comment, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (assignment_id, a["review_id"], aid, guideline_id, dialect_label, prejudice_label,
         prejudice_target, comment, now()))
    ann_id = cur.lastrowid
    conn.executemany("INSERT INTO span(annotation_id, layer, start_char, end_char, surface, subtype) "
                     "VALUES (?,?,?,?,?,?)", [(ann_id, s.layer, s.start, s.end, s.surface, s.subtype) for s in spans])
    conn.execute("UPDATE assignment SET status='done', completed_at=?, locked_until=NULL WHERE id=?",
                 (now(), assignment_id))
    conn.commit()
    return ann_id


def own_annotation(conn: sqlite3.Connection, alias: str, review_id: str) -> dict | None:
    """Unica lectura de anotaciones permitida a un anotador: la suya."""
    aid = _annotator_id(conn, alias)
    r = conn.execute("SELECT * FROM annotation WHERE annotator_id=? AND review_id=?", (aid, review_id)).fetchone()
    return dict(r) if r else None


def spans_of(conn: sqlite3.Connection, annotation_id: int) -> list[Span]:
    return [Span(r["start_char"], r["end_char"], r["layer"], r["subtype"], r["surface"])
            for r in conn.execute("SELECT * FROM span WHERE annotation_id=? ORDER BY start_char", (annotation_id,))]


def flag_disagreements(conn: sqlite3.Connection) -> int:
    """Crea tarea de ronda 2 (tercer anotador) cuando los dos de la ronda 1
    discrepan en la etiqueta dialectal o prejuiciosa, o alguno marco AMBIGUO."""
    annotators = [r[0] for r in conn.execute("SELECT id FROM annotator ORDER BY id")]
    rows = conn.execute(
        "SELECT review_id FROM annotation GROUP BY review_id HAVING COUNT(*)=2 "
        "AND review_id NOT IN (SELECT review_id FROM assignment WHERE round=2)").fetchall()
    n = 0
    for (rid,) in [tuple(r) for r in rows]:
        anns = conn.execute("SELECT annotator_id, dialect_label, prejudice_label FROM annotation WHERE review_id=?",
                            (rid,)).fetchall()
        d = {a["dialect_label"] for a in anns}
        p = {a["prejudice_label"] for a in anns}
        if len(d) > 1 or len(p) > 1 or "AMBIGUO" in d or "AMBIGUO" in p:
            used = {a["annotator_id"] for a in anns}
            cand = [x for x in annotators if x not in used]
            if not cand:
                continue
            load = {x: conn.execute("SELECT COUNT(*) FROM assignment WHERE annotator_id=? AND status!='done'",
                                    (x,)).fetchone()[0] for x in cand}
            third = min(cand, key=lambda x: (load[x], x))
            conn.execute("INSERT OR IGNORE INTO assignment(review_id, annotator_id, round, status, created_at) "
                         "VALUES (?,?,2,'pending',?)", (rid, third, now()))
            n += 1
    conn.commit()
    return n


def _majority(labels: list[str]) -> str | None:
    c = Counter(l for l in labels if l != "AMBIGUO")
    if not c:
        return None
    (lab, k), *rest = c.most_common()
    if k >= 2 and (not rest or rest[0][1] < k):
        return lab
    return None


def compute_gold(conn: sqlite3.Connection) -> dict:
    """Etiqueta final: mayoria (>=2 coinciden) entre las anotaciones completadas.
    Sin mayoria -> PENDIENTE_CONSENSO (decision humana en reunion de consenso)."""
    out = Counter()
    reviews = [r[0] for r in conn.execute("SELECT DISTINCT review_id FROM annotation")]
    for rid in reviews:
        anns = conn.execute("SELECT dialect_label, prejudice_label FROM annotation WHERE review_id=?", (rid,)).fetchall()
        if len(anns) < 2:
            continue
        dg = _majority([a["dialect_label"] for a in anns])
        pg = _majority([a["prejudice_label"] for a in anns])
        method = "mayoria" if len(anns) >= 3 else "acuerdo_2_de_2"
        if dg is None and len(anns) == 2 and anns[0]["dialect_label"] == anns[1]["dialect_label"] == "AMBIGUO":
            dg = "EXCLUIDO"
        dg = dg or "PENDIENTE_CONSENSO"
        pg = pg or "PENDIENTE_CONSENSO"
        conn.execute("INSERT OR REPLACE INTO gold_label(review_id, dialect_gold, prejudice_gold, method, "
                     "n_annotations, decided_at) VALUES (?,?,?,?,?,?)", (rid, dg, pg, method, len(anns), now()))
        out[dg] += 1
    conn.commit()
    return dict(out)
