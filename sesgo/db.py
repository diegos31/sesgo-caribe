"""Modelo de datos consolidado (SQLite) y acceso."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS corpus_source (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  version TEXT,
  access_note TEXT
);

CREATE TABLE IF NOT EXISTS ingest_batch (
  id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES corpus_source(id),
  filename TEXT NOT NULL,
  file_sha256 TEXT NOT NULL UNIQUE,
  n_read INTEGER, n_inserted INTEGER, n_duplicates INTEGER, n_rejected INTEGER,
  summary_json TEXT,
  status TEXT NOT NULL CHECK (status IN ('running','done','failed')),
  started_at TEXT NOT NULL, finished_at TEXT
);

-- Texto inmutable y sin datos de autor. 'country' = pais del LUGAR reseniado.
CREATE TABLE IF NOT EXISTS review (
  id TEXT PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES corpus_source(id),
  batch_id INTEGER NOT NULL REFERENCES ingest_batch(id),
  text_norm TEXT NOT NULL,
  text_sha256 TEXT NOT NULL UNIQUE,
  dedupe_key TEXT NOT NULL,
  polarity INTEGER NOT NULL CHECK (polarity BETWEEN 1 AND 5),
  y_class TEXT NOT NULL CHECK (y_class IN ('NEG','NEU','POS')),
  place_type TEXT,
  country TEXT,
  n_chars INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_review_strata ON review(country, y_class, place_type);
CREATE INDEX IF NOT EXISTS ix_review_dedupe ON review(dedupe_key);

CREATE TABLE IF NOT EXISTS annotator (
  id INTEGER PRIMARY KEY,
  alias TEXT NOT NULL UNIQUE,
  dialect_familiarity TEXT
);

CREATE TABLE IF NOT EXISTS guideline_version (
  id INTEGER PRIMARY KEY,
  version TEXT NOT NULL UNIQUE,
  file_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_draw (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  seed INTEGER NOT NULL,
  params_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_item (
  draw_id INTEGER NOT NULL REFERENCES sample_draw(id),
  review_id TEXT NOT NULL REFERENCES review(id),
  stratum TEXT NOT NULL,
  pool TEXT NOT NULL CHECK (pool IN ('random','lexicon')),
  lexicon_hits INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (draw_id, review_id)
);

-- Asignacion: cada (resenia, anotador, ronda) es una tarea independiente.
-- El bloqueo es por tarea y por anotador: no impide la segunda anotacion
-- independiente de otro anotador.
CREATE TABLE IF NOT EXISTS assignment (
  id INTEGER PRIMARY KEY,
  review_id TEXT NOT NULL REFERENCES review(id),
  annotator_id INTEGER NOT NULL REFERENCES annotator(id),
  round INTEGER NOT NULL CHECK (round IN (1,2)),
  status TEXT NOT NULL CHECK (status IN ('pending','locked','done')),
  locked_until TEXT,
  created_at TEXT NOT NULL, completed_at TEXT,
  UNIQUE (review_id, annotator_id, round)
);
CREATE INDEX IF NOT EXISTS ix_assignment_queue ON assignment(annotator_id, status);

CREATE TABLE IF NOT EXISTS annotation (
  id INTEGER PRIMARY KEY,
  assignment_id INTEGER NOT NULL UNIQUE REFERENCES assignment(id),
  review_id TEXT NOT NULL REFERENCES review(id),
  annotator_id INTEGER NOT NULL REFERENCES annotator(id),
  guideline_id INTEGER NOT NULL REFERENCES guideline_version(id),
  dialect_label TEXT NOT NULL CHECK (dialect_label IN ('MARCADO','NO_MARCADO','AMBIGUO')),
  prejudice_label TEXT NOT NULL CHECK (prejudice_label IN ('PRESENTE','AUSENTE','AMBIGUO')),
  prejudice_target TEXT,
  comment TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS span (
  id INTEGER PRIMARY KEY,
  annotation_id INTEGER NOT NULL REFERENCES annotation(id) ON DELETE CASCADE,
  layer TEXT NOT NULL CHECK (layer IN ('DIALECT','PREJ')),
  start_char INTEGER NOT NULL, end_char INTEGER NOT NULL,
  surface TEXT NOT NULL,
  subtype TEXT,
  CHECK (start_char >= 0 AND end_char > start_char)
);

CREATE TABLE IF NOT EXISTS gold_label (
  review_id TEXT PRIMARY KEY REFERENCES review(id),
  dialect_gold TEXT NOT NULL,
  prejudice_gold TEXT NOT NULL,
  method TEXT NOT NULL,
  n_annotations INTEGER NOT NULL,
  decided_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_split (
  version TEXT NOT NULL,
  review_id TEXT NOT NULL REFERENCES review(id),
  role TEXT NOT NULL CHECK (role IN ('train','dev','test','audit')),
  PRIMARY KEY (version, review_id)
);

CREATE TABLE IF NOT EXISTS model_run (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  backend TEXT NOT NULL,
  split_version TEXT NOT NULL,
  params_json TEXT NOT NULL,
  metrics_json TEXT,
  config_hash TEXT,
  started_at TEXT NOT NULL, finished_at TEXT
);

CREATE TABLE IF NOT EXISTS prediction (
  run_id INTEGER NOT NULL REFERENCES model_run(id),
  review_id TEXT NOT NULL REFERENCES review(id),
  y_pred TEXT NOT NULL CHECK (y_pred IN ('NEG','NEU','POS')),
  proba_json TEXT,
  PRIMARY KEY (run_id, review_id)
);

CREATE TABLE IF NOT EXISTS audit_result (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES model_run(id),
  scope TEXT NOT NULL,
  metric TEXT NOT NULL,
  estimate REAL, ci_low REAL, ci_high REAL,
  n_marked INTEGER, n_unmarked INTEGER,
  note TEXT,
  created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
    conn.commit()
