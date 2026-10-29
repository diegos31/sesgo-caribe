"""Particion train/dev/test + conjunto de auditoria (anotado, reservado).

- 'audit': resenias con etiqueta final de grupo (MARCADO / NO_MARCADO). Nunca se
  usan para entrenar ni ajustar. Se evaluan una sola vez con el protocolo congelado.
- 'train'/'dev'/'test': el resto del corpus, estratificado por clase y tipo de lugar.
- Anti-fuga: se retiran de train/dev/test los textos cuya clave de duplicado
  trivial coincide con alguno del conjunto de auditoria.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd


def make_splits(conn: sqlite3.Connection, version: str, seed: int, dev_frac: float, test_frac: float,
                audit_draw_ids: tuple[int, ...] | None = None) -> dict:
    if conn.execute("SELECT 1 FROM data_split WHERE version=? LIMIT 1", (version,)).fetchone():
        raise ValueError(f"La particion '{version}' ya existe.")
    df = pd.read_sql_query("SELECT id, y_class, place_type, dedupe_key FROM review", conn)
    gold = pd.read_sql_query(
        "SELECT review_id FROM gold_label WHERE dialect_gold IN ('MARCADO','NO_MARCADO')", conn)
    audit_ids = set(gold["review_id"])
    if audit_draw_ids:
        q = ",".join("?" * len(audit_draw_ids))
        in_draw = {r[0] for r in conn.execute(f"SELECT review_id FROM sample_item WHERE draw_id IN ({q})", audit_draw_ids)}
        audit_ids &= in_draw
    audit_keys = set(df[df.id.isin(audit_ids)]["dedupe_key"])
    rest = df[~df.id.isin(audit_ids)]
    leaked = rest[rest.dedupe_key.isin(audit_keys)]
    rest = rest[~rest.id.isin(set(leaked.id))]
    rng = np.random.RandomState(seed)
    roles: dict[str, str] = {i: "audit" for i in audit_ids}
    for _, part in rest.groupby(["y_class", rest["place_type"].fillna("NA")]):
        ids = part["id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n_dev = int(round(len(ids) * dev_frac))
        n_test = int(round(len(ids) * test_frac))
        for j, rid in enumerate(ids):
            roles[rid] = "dev" if j < n_dev else ("test" if j < n_dev + n_test else "train")
    conn.executemany("INSERT INTO data_split(version, review_id, role) VALUES (?,?,?)",
                     [(version, r, role) for r, role in roles.items()])
    conn.commit()
    counts = pd.Series(roles).value_counts().to_dict()
    return {"version": version, "conteos": counts, "retiradas_por_fuga": int(len(leaked))}


def load_role(conn: sqlite3.Connection, version: str, role: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT r.id, r.text_norm, r.polarity, r.y_class, r.place_type, r.n_chars FROM data_split s "
        "JOIN review r ON r.id=s.review_id WHERE s.version=? AND s.role=?", conn, params=(version, role))
