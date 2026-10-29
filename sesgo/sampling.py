"""Disponibilidad (viabilidad) y muestreo estratificado reproducible."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from .db import now
from .lexicon import LexEntry, compile_lexicon, count_hits


def load_reviews(conn: sqlite3.Connection, country: str | None = None) -> pd.DataFrame:
    q = "SELECT id, text_norm, polarity, y_class, place_type, country, n_chars FROM review"
    params: tuple = ()
    if country:
        q += " WHERE lower(country) = lower(?)"
        params = (country,)
    return pd.read_sql_query(q, conn, params=params)


def feasibility(conn: sqlite3.Connection, entries: list[LexEntry], country: str | None = None) -> pd.DataFrame:
    """Conteo por estrato y numero de candidatas con >=1 coincidencia lexica."""
    df = load_reviews(conn, country)
    pat = compile_lexicon(entries)
    df["lex_hits"] = [count_hits(t, pat) for t in df["text_norm"]]
    df["lex_candidate"] = df["lex_hits"] > 0
    g = (df.groupby(["country", "y_class", "place_type"], dropna=False)
           .agg(n_reviews=("id", "size"), n_lex_candidates=("lex_candidate", "sum"))
           .reset_index())
    return g


def draw_sample(conn: sqlite3.Connection, name: str, seed: int, n_random: int, n_lexicon: int,
                entries: list[LexEntry], country: str | None = None,
                exclude_draw_ids: tuple[int, ...] = ()) -> dict:
    """Por estrato (y_class x place_type): `n_random` al azar de todo el estrato
    (sirve para estimar prevalencia sin sesgo del lexico) y `n_lexicon` extra
    entre candidatas lexicas aun no elegidas (sobremuestreo del grupo raro)."""
    df = load_reviews(conn, country)
    if exclude_draw_ids:
        q = ",".join("?" * len(exclude_draw_ids))
        ex = {r[0] for r in conn.execute(f"SELECT review_id FROM sample_item WHERE draw_id IN ({q})", exclude_draw_ids)}
        df = df[~df["id"].isin(ex)]
    pat = compile_lexicon(entries)
    df["lex_hits"] = [count_hits(t, pat) for t in df["text_norm"]]
    rng = np.random.RandomState(seed)
    chosen: list[tuple[str, str, str, int]] = []
    shortfalls: dict[str, dict] = {}
    for (yc, pt), part in df.groupby(["y_class", df["place_type"].fillna("NA")]):
        stratum = f"{yc}|{pt}"
        ids = part["id"].to_numpy()
        k = min(n_random, len(ids))
        pick = rng.choice(ids, size=k, replace=False) if k else np.array([], dtype=object)
        hits = dict(zip(part["id"], part["lex_hits"]))
        for rid in pick:
            chosen.append((rid, stratum, "random", int(hits[rid])))
        rest = part[(part["lex_hits"] > 0) & (~part["id"].isin(set(pick)))]["id"].to_numpy()
        k2 = min(n_lexicon, len(rest))
        pick2 = rng.choice(rest, size=k2, replace=False) if k2 else np.array([], dtype=object)
        for rid in pick2:
            chosen.append((rid, stratum, "lexicon", int(hits[rid])))
        if k < n_random or k2 < n_lexicon:
            shortfalls[stratum] = {"random": [k, n_random], "lexicon": [k2, n_lexicon]}
    params = {"country": country, "n_random": n_random, "n_lexicon": n_lexicon,
              "n_items": len(chosen), "shortfalls": shortfalls}
    cur = conn.execute("INSERT INTO sample_draw(name, seed, params_json, created_at) VALUES (?,?,?,?)",
                       (name, seed, json.dumps(params, ensure_ascii=False), now()))
    draw_id = cur.lastrowid
    conn.executemany("INSERT INTO sample_item(draw_id, review_id, stratum, pool, lexicon_hits) VALUES (?,?,?,?,?)",
                     [(draw_id, r, s, p, h) for r, s, p, h in chosen])
    conn.commit()
    return {"draw_id": draw_id, **params}
