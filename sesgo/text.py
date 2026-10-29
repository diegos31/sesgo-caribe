"""Normalizacion, reduccion de datos personales y tokenizacion.

Decision de diseno: NO se quitan tildes ni se corrige ortografia, porque eso
destruiria justamente las senales dialectales/ortograficas que se quieren
estudiar. Solo se eliminan caracteres de control, se unifican espacios y se
reemplazan datos de contacto por marcadores.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_MENTION = re.compile(r"(?<!\w)@\w{2,}")
_PHONE_CAND = re.compile(r"(?<!\w)\+?\d[\d\s().-]{5,}\d(?!\w)")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+['\u2019]?|[^\w\s]")


def _redact_phone(m: re.Match) -> str:
    digits = sum(ch.isdigit() for ch in m.group(0))
    return "[TEL]" if 7 <= digits <= 15 else m.group(0)


def normalize_text(raw: str) -> tuple[str, dict]:
    """Devuelve (texto_normalizado, conteo_de_redacciones)."""
    flags = {"url": 0, "email": 0, "mention": 0, "phone": 0}
    t = unicodedata.normalize("NFC", raw or "")
    t = _CTRL.sub("", t)
    t, flags["url"] = _URL.subn("[URL]", t)
    t, flags["email"] = _EMAIL.subn("[EMAIL]", t)
    t, flags["mention"] = _MENTION.subn("[USUARIO]", t)
    before = t
    t = _PHONE_CAND.sub(_redact_phone, t)
    flags["phone"] = t.count("[TEL]") - before.count("[TEL]")
    t = _WS.sub(" ", t).strip()
    return t, flags


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedupe_key(text: str) -> str:
    """Clave para detectar duplicados triviales (mayusculas, signos, espacios)."""
    core = "".join(ch for ch in text.casefold() if ch.isalnum())
    return hashlib.sha1(core.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Tokenizacion por expresion regular con desplazamientos de caracter.

    Una palabra con apostrofe final (p. ej. "pa'") es un solo token.
    """
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]
