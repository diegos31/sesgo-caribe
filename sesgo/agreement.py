"""Concordancia entre anotadores (Cohen, Fleiss) a nivel de etiqueta y de token."""
from __future__ import annotations

import itertools
import sqlite3

import numpy as np
import pandas as pd

from .spans import spans_to_bio, snap_to_tokens
from .annotation import spans_of


def percent_agreement(a, b) -> float:
    a, b = list(a), list(b)
    return float(np.mean([x == y for x, y in zip(a, b)])) if a else float("nan")


def cohen_kappa(a, b) -> float:
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def fleiss_kappa(table: np.ndarray) -> float:
    """table[i, j] = n.o de anotadores que asignaron la categoria j al item i
    (mismo numero de anotadores por item)."""
    table = np.asarray(table, dtype=float)
    N, _ = table.shape
    n = table[0].sum()
    p_j = table.sum(axis=0) / (N * n)
    P_i = (np.square(table).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), np.square(p_j).sum()
    return float("nan") if P_e == 1 else float((P_bar - P_e) / (1 - P_e))


def interpret_kappa(k: float) -> str:
    """Escala de Landis y Koch (1977)."""
    if np.isnan(k):
        return "indefinido"
    for thr, lab in [(0.0, "pobre"), (0.20, "leve"), (0.40, "aceptable"),
                     (0.60, "moderado"), (0.80, "sustancial"), (1.01, "casi perfecto")]:
        if k < thr or (thr == 1.01 and k <= thr):
            return lab
    return "casi perfecto"


def _round1(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT an.id AS annotation_id, an.review_id, a.alias, an.dialect_label, an.prejudice_label "
        "FROM annotation an JOIN annotator a ON a.id=an.annotator_id "
        "JOIN assignment s ON s.id=an.assignment_id WHERE s.round=1", conn)


def pairwise_label_agreement(conn: sqlite3.Connection, field: str = "dialect_label") -> pd.DataFrame:
    df = _round1(conn)
    rows = []
    for x, y in itertools.combinations(sorted(df["alias"].unique()), 2):
        m = (df[df.alias == x][["review_id", field]].merge(df[df.alias == y][["review_id", field]],
                                                           on="review_id", suffixes=("_a", "_b")))
        rows.append({"anotador_a": x, "anotador_b": y, "campo": field, "n_comunes": len(m),
                     "acuerdo_observado": percent_agreement(m[field + "_a"], m[field + "_b"]),
                     "kappa_cohen": cohen_kappa(m[field + "_a"], m[field + "_b"])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["interpretacion"] = out["kappa_cohen"].map(interpret_kappa)
    return out


def fleiss_for_all(conn: sqlite3.Connection, field: str = "dialect_label") -> dict:
    """Fleiss sobre las resenias etiquetadas por exactamente 3 anotadores."""
    df = pd.read_sql_query(f"SELECT review_id, {field} AS lab FROM annotation", conn)
    cnt = df.groupby("review_id").size()
    ids = cnt[cnt == 3].index
    if len(ids) == 0:
        return {"n_items": 0, "kappa_fleiss": float("nan")}
    sub = df[df.review_id.isin(ids)]
    tab = pd.crosstab(sub.review_id, sub.lab).to_numpy()
    k = fleiss_kappa(tab)
    return {"n_items": int(len(ids)), "kappa_fleiss": k, "interpretacion": interpret_kappa(k)}


def token_level_agreement(conn: sqlite3.Connection, layer: str = "DIALECT") -> pd.DataFrame:
    """Kappa de Cohen sobre etiquetas BIO por token (ronda 1, pares de anotadores)."""
    df = _round1(conn)
    texts = dict(conn.execute("SELECT id, text_norm FROM review").fetchall())
    per: dict[tuple[str, str], tuple[list, list]] = {}
    for rid, g in df.groupby("review_id"):
        if len(g) != 2:
            continue
        tags = []
        for _, r in g.iterrows():
            sp = [snap_to_tokens(texts[rid], s) for s in spans_of(conn, int(r.annotation_id))]
            tags.append([t[3] for t in spans_to_bio(texts[rid], sp, layer, strict=False)])
        key = tuple(sorted(g["alias"]))
        per.setdefault(key, ([], []))
        order = np.argsort(g["alias"].to_numpy())
        per[key][0].extend(tags[order[0]])
        per[key][1].extend(tags[order[1]])
    rows = [{"anotador_a": k[0], "anotador_b": k[1], "capa": layer, "n_tokens": len(v[0]),
             "kappa_token_bio": cohen_kappa(v[0], v[1])} for k, v in sorted(per.items())]
    return pd.DataFrame(rows)


def fleiss_pilot(conn: sqlite3.Connection, field: str, pilot_name: str) -> dict:
    """Fleiss solo sobre el sorteo piloto: los 3 anotadores etiquetan el MISMO lote
    sin seleccion por discrepancia (un Fleiss sobre casos discordantes estaria sesgado)."""
    df = pd.read_sql_query(
        f"SELECT an.review_id, an.{field} AS lab FROM annotation an "
        "JOIN sample_item si ON si.review_id=an.review_id JOIN sample_draw sd ON sd.id=si.draw_id "
        "WHERE sd.name=?", conn, params=(pilot_name,))
    cnt = df.groupby("review_id").size()
    ids = cnt[cnt == 3].index
    if len(ids) == 0:
        return {"n_items": 0, "kappa_fleiss": float("nan")}
    tab = pd.crosstab(df[df.review_id.isin(ids)].review_id, df[df.review_id.isin(ids)].lab).to_numpy()
    k = fleiss_kappa(tab)
    return {"n_items": int(len(ids)), "kappa_fleiss": k, "interpretacion": interpret_kappa(k)}
