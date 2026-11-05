"""Tests for the staged SQLAlchemy database connection foundation."""

from sqlalchemy import text

from src.database.sqlalchemy_support import (
    create_database_engine,
    create_session_factory,
    configured_database_url,
)


def test_database_url_defaults_to_existing_sqlite_location(monkeypatch):
    monkeypatch.delenv("NARS_DATABASE_URL", raising=False)
    assert configured_database_url().startswith("sqlite:///")


def test_database_url_can_be_configured_for_postgresql(monkeypatch):
    url = "postgresql+psycopg://nars:example@localhost:5432/nars"
    monkeypatch.setenv("NARS_DATABASE_URL", url)
    assert configured_database_url() == url


def test_sqlalchemy_sqlite_engine_and_session_factory(tmp_path):
    path = tmp_path / "orm-foundation.db"
    engine = create_database_engine("sqlite:///" + path.as_posix())
    sessions = create_session_factory(engine)

    with sessions() as session:
        assert session.execute(text("SELECT 1")).scalar_one() == 1

    engine.dispose()
