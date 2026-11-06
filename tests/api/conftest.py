"""Fixtures for API tests using an isolated temporary SQLite database."""

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_current_user
from src.api.main import app
from src.database import db


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    database_path = tmp_path / "nars-api-test.db"
    database_url = "sqlite:///" + database_path.as_posix()
    monkeypatch.setattr(db, "DB_PATH", database_path)
    monkeypatch.setenv("NARS_DATABASE_URL", database_url)
    monkeypatch.setenv("NARS_JWT_SECRET", "test-only-secret-key-with-at-least-32-bytes")
    db.initialize_database()

    # Existing functional API tests focus on their endpoint behavior.
    # Authentication itself is exercised in test_auth.py.
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": 1,
        "username": "test-admin",
        "role": "admin",
        "is_active": True,
    }
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        engine = getattr(app.state, "database_engine", None)
        if engine is not None:
            engine.dispose()
            del app.state.database_engine
        if hasattr(app.state, "database_session_factory"):
            del app.state.database_session_factory
