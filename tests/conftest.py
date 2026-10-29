import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def conn(tmp_path):
    from sesgo import db
    c = db.connect(tmp_path / "t.db")
    db.init_db(c)
    return c


@pytest.fixture()
def loaded(conn):
    from sesgo import ingest
    ingest.ingest_file(conn, ROOT / "data" / "fixture" / "reviews_fixture.csv", "FIXTURE_SINTETICO", country_filter="Colombia")
    return conn
