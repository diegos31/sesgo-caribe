"""Anotacion de tramos (stand-off) y conversion determinista a etiquetas BIO.

Representacion CANONICA: tramos [start, end) en caracteres sobre `review.text_norm`
(inmutable). Las etiquetas BIO por token se DERIVAN con `spans_to_bio` usando el
tokenizador de `sesgo.text.tokenize`; la conversion inversa es `bio_to_spans`.
Cada capa (DIALECT, PREJ) tiene su propio esquema BIO independiente: asi un mismo
token puede ser a la vez rasgo dialectal y parte de una expresion prejuiciosa.
"""
from __future__ import annotations

from dataclasses import dataclass

from .text import tokenize

LAYERS = {"DIALECT": "DIAL", "PREJ": "PREJ"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    layer: str
    subtype: str | None = None
    surface: str | None = None


def validate_spans(text: str, spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for s in spans:
        if s.layer not in LAYERS:
            raise ValueError(f"Capa desconocida: {s.layer}")
        if not (0 <= s.start < s.end <= len(text)):
            raise ValueError(f"Tramo fuera de rango: {s}")
        surface = text[s.start:s.end]
        if s.surface is not None and s.surface != surface:
            raise ValueError(f"El texto del tramo no coincide: {s.surface!r} != {surface!r}")
        out.append(Span(s.start, s.end, s.layer, s.subtype, surface))
    for layer in LAYERS:
        ls = sorted((s for s in out if s.layer == layer), key=lambda x: x.start)
        for a, b in zip(ls, ls[1:]):
            if b.start < a.end:
                raise ValueError(f"Tramos solapados en la capa {layer}: {a} y {b}")
    return out


def snap_to_tokens(text: str, span: Span) -> Span:
    toks = tokenize(text)
    cov = [t for t in toks if t[1] < span.end and t[2] > span.start]
    if not cov:
        raise ValueError(f"El tramo no cubre ningun token: {span}")
    st, en = cov[0][1], cov[-1][2]
    return Span(st, en, span.layer, span.subtype, text[st:en])


def spans_to_bio(text: str, spans: list[Span], layer: str, strict: bool = True):
    """Lista de (token, start, end, etiqueta). Si strict, los tramos deben
    coincidir con limites de token (use `snap_to_tokens` en la interfaz)."""
    toks = tokenize(text)
    tags = ["O"] * len(toks)
    prefix = LAYERS[layer]
    for s in sorted((x for x in spans if x.layer == layer), key=lambda x: x.start):
        cov = [i for i, (_, a, b) in enumerate(toks) if a < s.end and b > s.start]
        if not cov:
            raise ValueError(f"El tramo no cubre ningun token: {s}")
        if strict and (toks[cov[0]][1] != s.start or toks[cov[-1]][2] != s.end):
            raise ValueError(f"El tramo {s} no coincide con limites de token")
        tags[cov[0]] = f"B-{prefix}"
        for i in cov[1:]:
            tags[i] = f"I-{prefix}"
    return [(t, a, b, tag) for (t, a, b), tag in zip(toks, tags)]


def bio_to_spans(text: str, bio, layer: str, subtype: str | None = None) -> list[Span]:
    prefix = LAYERS[layer]
    spans: list[Span] = []
    cur: list[int] | None = None
    for _, a, b, tag in bio:
        if tag == f"B-{prefix}":
            if cur:
                spans.append(Span(cur[0], cur[1], layer, subtype, text[cur[0]:cur[1]]))
            cur = [a, b]
        elif tag == f"I-{prefix}":
            if cur is None:
                raise ValueError("Etiqueta I sin B previa")
            cur[1] = b
        else:
            if cur:
                spans.append(Span(cur[0], cur[1], layer, subtype, text[cur[0]:cur[1]]))
                cur = None
    if cur:
        spans.append(Span(cur[0], cur[1], layer, subtype, text[cur[0]:cur[1]]))
    return spans
