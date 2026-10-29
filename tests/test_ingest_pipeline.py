import pytest
from sesgo import ingest


def test_ingest_counts_dedup_and_drops_author_columns(conn):
    from tests.conftest import ROOT
    r = ingest.ingest_file(conn, ROOT / "data/fixture/reviews_fixture.csv", "FIXTURE_SINTETICO", country_filter="Colombia")
    assert r["leidas"] == 1010 and r["duplicadas"] > 0
    assert set(r["columnas_descartadas"]) == {"Town", "UserName"}
    cols = [c[1] for c in conn.execute("PRAGMA table_info(review)")]
    assert "UserName" not in cols and "author" not in " ".join(cols).lower()
    assert conn.execute("SELECT COUNT(*) FROM review WHERE text_norm LIKE '%@%'").fetchone()[0] == 0
    assert r["redacciones"]["email"] > 0


def test_ingest_is_idempotent(loaded):
    from tests.conftest import ROOT
    with pytest.raises(ValueError):
        ingest.ingest_file(loaded, ROOT / "data/fixture/reviews_fixture.csv", "FIXTURE_SINTETICO")


def test_polarity_class_mapping(loaded):
    rows = loaded.execute("SELECT polarity, y_class FROM review").fetchall()
    m = {1: "NEG", 2: "NEG", 3: "NEU", 4: "POS", 5: "POS"}
    assert all(m[r["polarity"]] == r["y_class"] for r in rows)
