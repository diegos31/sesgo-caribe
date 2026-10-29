import pytest
from sesgo.spans import Span, bio_to_spans, snap_to_tokens, spans_to_bio, validate_spans
from sesgo.text import normalize_text, tokenize


def test_normalize_redacts_and_keeps_accents():
    t, f = normalize_text("Llamen +57 300 123 4567 o a x@y.com  @juan http://a.b   Qué rico, mi pelao")
    assert "[TEL]" in t and "[EMAIL]" in t and "[USUARIO]" in t and "[URL]" in t
    assert "Qué" in t and "  " not in t and f["phone"] == 1


def test_short_numbers_are_not_phones():
    t, f = normalize_text("Pagamos 45000 pesos por 3 noches")
    assert f["phone"] == 0 and "45000" in t


def test_apostrophe_token():
    assert [x[0] for x in tokenize("lo' niño' van pa' allá")][:3] == ["lo'", "niño'", "van"]


def test_bio_roundtrip():
    text = "El plato estuvo muy zumbarro y chiriguel de verdad."
    a, b = text.index("zumbarro"), text.index("chiriguel")
    sp = [Span(a, a + 8, "DIALECT"), Span(b, b + 9, "DIALECT")]
    sp = [snap_to_tokens(text, s) for s in sp]
    sp = validate_spans(text, sp)
    bio = spans_to_bio(text, sp, "DIALECT")
    back = bio_to_spans(text, bio, "DIALECT")
    assert [(s.start, s.end) for s in back] == [(s.start, s.end) for s in sp]


def test_adjacent_spans_stay_separate_and_multiword_spans_merge():
    text = "no estaba nada bien"
    sp = [Span(0, 9, "PREJ"), Span(10, 14, "PREJ")]
    bio = spans_to_bio(text, sp, "PREJ")
    assert [b[3] for b in bio] == ["B-PREJ", "I-PREJ", "B-PREJ", "O"]
    assert len(bio_to_spans(text, bio, "PREJ")) == 2


def test_layers_are_independent_and_overlap_across_layers_is_allowed():
    text = "Son unos bruzaco"
    sp = validate_spans(text, [Span(9, 16, "DIALECT"), Span(9, 16, "PREJ")])
    assert spans_to_bio(text, sp, "DIALECT")[-1][3] == "B-DIAL"
    assert spans_to_bio(text, sp, "PREJ")[-1][3] == "B-PREJ"


def test_validation_errors():
    with pytest.raises(ValueError):
        validate_spans("hola mundo", [Span(0, 4, "DIALECT"), Span(2, 6, "DIALECT")])
    with pytest.raises(ValueError):
        validate_spans("hola", [Span(0, 9, "DIALECT")])
    with pytest.raises(ValueError):
        spans_to_bio("hola mundo", [Span(1, 4, "DIALECT")], "DIALECT")  # no alineado
