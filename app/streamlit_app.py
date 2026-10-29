"""Interfaz Streamlit (una sola aplicacion; llama directamente a la capa de servicio)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sesgo import annotation, db, fairness, ingest  # noqa: E402
from sesgo.annotation import PREJ_TARGETS  # noqa: E402
from sesgo.spans import Span  # noqa: E402
from sesgo.text import tokenize  # noqa: E402

DB_PATH = os.environ.get("SESGO_DB", str(Path(__file__).resolve().parent.parent / "work" / "demo.db"))
conn = db.connect(DB_PATH)
db.init_db(conn)

st.set_page_config(page_title="Auditoria de sesgo lingüístico", layout="wide")
page = st.sidebar.radio("Navegación", ["Panel", "Ingesta", "Anotación", "Auditoría"])
st.sidebar.caption(f"BD: {Path(DB_PATH).name}")


def _spans_from_idx(toks, idx, layer):
    spans, run = [], []
    for i in sorted(idx):
        if run and i != run[-1] + 1:
            spans.append(run); run = []
        run.append(i)
    if run:
        spans.append(run)
    return [Span(toks[r[0]][1], toks[r[-1]][2], layer) for r in spans]


if page == "Panel":
    st.title("Panel general")
    q = lambda s: conn.execute(s).fetchone()[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reseñas", q("SELECT COUNT(*) FROM review"))
    c2.metric("Tareas hechas", q("SELECT COUNT(*) FROM assignment WHERE status='done'"))
    c3.metric("Etiquetas finales", q("SELECT COUNT(*) FROM gold_label"))
    c4.metric("Corridas de modelo", q("SELECT COUNT(*) FROM model_run"))
    df = pd.read_sql_query("SELECT y_class AS clase, COUNT(*) AS n FROM review GROUP BY y_class", conn)
    if not df.empty:
        st.bar_chart(df.set_index("clase"))

elif page == "Ingesta":
    st.title("Ingesta de lote")
    up = st.file_uploader("Archivo CSV / TSV / JSONL", type=["csv", "tsv", "jsonl"])
    src = st.text_input("Nombre de la fuente", "RestMex2023")
    country = st.text_input("Filtrar por país del lugar reseñado (opcional)", "")
    if up is not None and st.button("Ingerir"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
            tmp.write(up.getvalue())
        try:
            st.json(ingest.ingest_file(conn, tmp.name, src, country_filter=country or None))
        except ValueError as e:
            st.error(str(e))

elif page == "Anotación":
    st.title("Anotación")
    aliases = [r[0] for r in conn.execute("SELECT alias FROM annotator ORDER BY alias")]
    if not aliases:
        st.info("No hay anotadores registrados.")
    else:
        alias = st.selectbox("Anotador", aliases)
        gl = conn.execute("SELECT id FROM guideline_version ORDER BY id DESC LIMIT 1").fetchone()
        task = annotation.next_task(conn, alias)
        if task is None:
            st.success("No tienes tareas pendientes.")
        elif gl is None:
            st.error("Falta registrar la versión de la guía de anotación.")
        else:
            toks = tokenize(task["text"])
            st.caption(f"Tarea {task['assignment_id']} · ronda {task['round']} · bloqueada hasta {task['locked_until']}")
            st.markdown(f"> {task['text']}")
            opts = [f"{i}: {t[0]}" for i, t in enumerate(toks)]
            d_sel = st.multiselect("Tokens con rasgo dialectal (capa A)", opts, key=f"d{task['assignment_id']}")
            dl = st.radio("Etiqueta dialectal", annotation.DIALECT_LABELS, horizontal=True, key=f"dl{task['assignment_id']}")
            p_sel = st.multiselect("Tokens de expresión prejuiciosa (capa B)", opts, key=f"p{task['assignment_id']}")
            pl = st.radio("Etiqueta prejuicio", annotation.PREJ_LABELS, horizontal=True, key=f"pl{task['assignment_id']}")
            tg = st.selectbox("Objetivo (solo si PRESENTE)", PREJ_TARGETS, key=f"t{task['assignment_id']}")
            if st.button("Guardar y continuar"):
                idx = lambda sel: [int(s.split(":")[0]) for s in sel]
                spans = _spans_from_idx(toks, idx(d_sel), "DIALECT") + _spans_from_idx(toks, idx(p_sel), "PREJ")
                try:
                    annotation.submit_annotation(conn, task["assignment_id"], alias, gl[0], dl, pl, spans,
                                                 prejudice_target=tg if pl == "PRESENTE" else None)
                    st.success("Guardado. Recarga para la siguiente reseña.")
                except (ValueError, PermissionError) as e:
                    st.error(str(e))

else:
    st.title("Auditoría de desempeño por grupo")
    runs = pd.read_sql_query("SELECT id, name, backend, split_version FROM model_run ORDER BY id", conn)
    if runs.empty:
        st.info("Aún no hay corridas de modelo.")
    else:
        st.dataframe(runs, use_container_width=True)
        rid = st.selectbox("Corrida", runs["id"].tolist())
        res = pd.read_sql_query("SELECT metric AS métrica, estimate AS brecha, ci_low AS ic_inf, ci_high AS ic_sup, "
                                "n_marked, n_unmarked, note FROM audit_result WHERE run_id=?", conn, params=(int(rid),))
        if res.empty:
            st.warning("Esta corrida aún no tiene auditoría (use `python -m sesgo audit`).")
        else:
            st.caption("Brecha D = NO_MARCADO − MARCADO (reponderado). D>0: peor desempeño en textos marcados.")
            st.dataframe(res, use_container_width=True)
            st.bar_chart(res.set_index("métrica")[["brecha"]])
