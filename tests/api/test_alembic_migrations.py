"""Smoke-test the versioned Alembic schema against disposable SQLite."""

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def test_alembic_upgrade_head_creates_versioned_nars_schema(tmp_path, monkeypatch):
    database = tmp_path / "alembic-smoke.db"
    monkeypatch.setenv("NARS_DATABASE_URL", "sqlite:///" + database.as_posix())

    config = Config("alembic.ini")
    command.upgrade(config, "head")

    engine = create_engine("sqlite:///" + database.as_posix())
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    assert {
        "alembic_version",
        "patients",
        "affected_areas",
        "clinical_assessments",
        "exercise_recommendations",
        "sessions",
        "session_frames",
        "adaptation_decisions",
        "api_users",
    }.issubset(tables)

    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
    assert version == "0002_api_users"

    api_user_indexes = {
        index["name"] for index in inspector.get_indexes("api_users")
    }
    assert "uq_api_users_username_lower" in api_user_indexes
    engine.dispose()
