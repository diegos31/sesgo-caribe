"""Ingesta por lotes de archivos estructurados (CSV/TSV/JSONL).

- Solo se leen las columnas del mapeo; TODAS las demas se descartan sin leerlas
  a la base (no se almacenan metadatos de autor).
- Idempotente: el SHA-256 del archivo identifica el lote.
- Los nombres de columna por defecto son un supuesto: deben confirmarse contra
  el archivo oficial de Rest-Mex y ajustarse con --map si difieren.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterator

from .config import polarity_to_class
from .db import now
from .text import dedupe_key, normalize_text, text_hash

DEFAULT_MAP = {"title": "Title", "text": "Review", "polarity": "Polarity",
               "type": "Type", "country": "Country"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_rows(path: Path) -> tuple[list[str], Iterator[dict]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        def gen():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
        first = next(gen(), {})
        return list(first.keys()), gen()
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    fh = open(path, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(fh, delimiter=delim)
    return list(reader.fieldnames or []), reader


def ingest_file(conn: sqlite3.Connection, path: str | Path, source_name: str,
                colmap: dict | None = None, country_filter: str | None = None,
                min_chars: int = 20, join_title: bool = True,
                source_note: str = "") -> dict:
    path = Path(path)
    colmap = {**DEFAULT_MAP, **(colmap or {})}
    sha = file_sha256(path)
    if conn.execute("SELECT 1 FROM ingest_batch WHERE file_sha256=?", (sha,)).fetchone():
        raise ValueError("Este archivo ya fue ingerido (mismo SHA-256).")
    conn.execute("INSERT OR IGNORE INTO corpus_source(name, access_note) VALUES (?,?)",
                 (source_name, source_note))
    source_id = conn.execute("SELECT id FROM corpus_source WHERE name=?", (source_name,)).fetchone()[0]
    cur = conn.execute(
        "INSERT INTO ingest_batch(source_id, filename, file_sha256, status, started_at) "
        "VALUES (?,?,?,?,?)", (source_id, path.name, sha, "running", now()))
    batch_id = cur.lastrowid

    fields, rows = iter_rows(path)
    used = {v for v in colmap.values() if v in fields}
    dropped = [f for f in fields if f not in used]
    rejected: Counter = Counter()
    redactions: Counter = Counter()
    n_read = n_ins = n_dup = 0
    try:
        for row in rows:
            n_read += 1
            try:
                pol = int(float(str(row.get(colmap["polarity"], "")).strip()))
                y_class = polarity_to_class(pol)
            except (ValueError, TypeError):
                rejected["polaridad_invalida"] += 1
                continue
            country = (row.get(colmap["country"]) or "").strip() or None
            if country_filter and (country or "").casefold() != country_filter.casefold():
                rejected["fuera_de_filtro_pais"] += 1
                continue
            body = (row.get(colmap["text"]) or "").strip()
            title = (row.get(colmap["title"]) or "").strip()
            raw = f"{title}. {body}" if (join_title and title) else body
            text, flags = normalize_text(raw)
            if len(text) < min_chars:
                rejected["texto_muy_corto"] += 1
                continue
            th = text_hash(text)
            ptype = (row.get(colmap["type"]) or "").strip().title() or None
            c = conn.execute(
                "INSERT OR IGNORE INTO review(id, source_id, batch_id, text_norm, text_sha256, "
                "dedupe_key, polarity, y_class, place_type, country, n_chars) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (th[:16], source_id, batch_id, text, th, dedupe_key(text), pol, y_class,
                 ptype, country, len(text)))
            if c.rowcount == 0:
                n_dup += 1
            else:
                n_ins += 1
                for k, v in flags.items():
                    redactions[k] += v
        summary = {"columnas_descartadas": dropped, "rechazos": dict(rejected),
                   "redacciones": dict(redactions)}
        conn.execute(
            "UPDATE ingest_batch SET n_read=?, n_inserted=?, n_duplicates=?, n_rejected=?, "
            "summary_json=?, status='done', finished_at=? WHERE id=?",
            (n_read, n_ins, n_dup, sum(rejected.values()),
             json.dumps(summary, ensure_ascii=False), now(), batch_id))
        conn.commit()
    except Exception:
        conn.execute("UPDATE ingest_batch SET status='failed', finished_at=? WHERE id=?",
                     (now(), batch_id))
        conn.commit()
        raise
    return {"batch_id": batch_id, "leidas": n_read, "insertadas": n_ins, "duplicadas": n_dup,
            "rechazadas": dict(rejected), "redacciones": dict(redactions),
            "columnas_descartadas": dropped}
