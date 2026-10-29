"""Configuracion y constantes del experimento."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "work" / "sesgo.db"
CLASSES = ("NEG", "NEU", "POS")
MARCADO = "MARCADO"
NO_MARCADO = "NO_MARCADO"


def polarity_to_class(p: int) -> str:
    """Transformacion 1-5 -> clase (definida en el diseno experimental).

    1-2 -> NEG, 3 -> NEU, 4-5 -> POS.
    """
    if p in (1, 2):
        return "NEG"
    if p == 3:
        return "NEU"
    if p in (4, 5):
        return "POS"
    raise ValueError(f"Polaridad fuera de rango 1-5: {p!r}")


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "experimento.json"
    return json.loads(path.read_text(encoding="utf-8"))


def config_hash(cfg: dict) -> str:
    canon = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()[:16]
