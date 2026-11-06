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
        "live_sessions",
        "patient_therapist_assignments",
    }.issubset(tables)

    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
    assert version == "0004_patient_assign"

    with engine.connect() as connection:
        username_index_sql = connection.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'index' AND name = 'uq_api_users_username_lower'"
            )
        ).scalar_one()
    assert "lower(username)" in username_index_sql
    engine.dispose()
