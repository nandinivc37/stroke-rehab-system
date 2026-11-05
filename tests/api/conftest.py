"""Fixtures for API tests using an isolated temporary SQLite database."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.database import db


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    database_path = tmp_path / "nars-api-test.db"
    monkeypatch.setattr(db, "DB_PATH", database_path)
    db.initialize_database()
    with TestClient(app) as client:
        yield client
