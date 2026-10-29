"""Lexico de apoyo para PRE-SELECCION de candidatas (no define el grupo).

El grupo MARCADO lo decide la anotacion humana. El lexico solo sirve para
sobremuestrear resenias con posibles marcas y para estimar disponibilidad.
Cada entrada debe citar una fuente lexicografica verificable (columna `source`).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LexEntry:
    term: str
    category: str
    source: str
    entry_url: str
    status: str  # verified | pending | fixture


def load_lexicon(path: str | Path, statuses: tuple[str, ...] | None = None) -> list[LexEntry]:
    out: list[LexEntry] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("term") or "").strip():
                continue
            e = LexEntry(row["term"].strip(), row.get("category", "").strip(),
                         row.get("source", "").strip(), row.get("entry_url", "").strip(),
                         row.get("status", "pending").strip())
            if statuses is None or e.status in statuses:
                out.append(e)
    return out


def _pattern(entries: list[LexEntry]) -> re.Pattern | None:
    if not entries:
        return None
    alts = sorted({re.escape(e.term) for e in entries}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE)


def find_hits(text: str, entries: list[LexEntry]) -> list[tuple[int, int, str]]:
    pat = _pattern(entries)
    if pat is None:
        return []
    return [(m.start(), m.end(), m.group(0)) for m in pat.finditer(text)]


def count_hits(text: str, pat: re.Pattern | None) -> int:
    return 0 if pat is None else len(pat.findall(text))


def compile_lexicon(entries: list[LexEntry]) -> re.Pattern | None:
    return _pattern(entries)
