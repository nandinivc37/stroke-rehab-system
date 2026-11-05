"""Safety tests for the SQLite-to-PostgreSQL migration preflight."""

import sqlite3

import pytest
from sqlalchemy import create_engine, text

from src.database.models import Base
from src.database.migrate_sqlite_to_postgres import (
    DataMigrationError,
    _snapshot_source,
    build_migration_plan,
)


def _make_source(path):
    engine = create_engine("sqlite:///" + path.as_posix())
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO patients(patient_id, name) VALUES (101, 'Synthetic Patient')")
        )
        connection.execute(text("PRAGMA user_version = 3"))
    engine.dispose()


def _make_target(path):
    engine = create_engine("sqlite:///" + path.as_posix())
    Base.metadata.create_all(engine)
    return engine


def test_migration_plan_is_read_only_and_reports_source_counts(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _make_source(source)
    engine = _make_target(target)

    plan = build_migration_plan(source, engine)

    assert plan["target_is_empty"] is True
    assert plan["row_counts"]["patients"] == 1
    assert plan["tables_to_copy"] == [
        "patients",
        "affected_areas",
        "clinical_assessments",
        "exercise_recommendations",
        "sessions",
        "session_frames",
        "adaptation_decisions",
    ]
    with sqlite3.connect(source) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM patients"
        ).fetchone()[0] == 1
    with engine.connect() as connection:
        assert connection.execute(
            text("SELECT COUNT(*) FROM patients")
        ).scalar_one() == 0
    engine.dispose()


def test_preflight_refuses_nonempty_target(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _make_source(source)
    engine = _make_target(target)
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO patients(patient_id, name) VALUES (999, 'Existing')")
        )

    with pytest.raises(DataMigrationError, match="must be empty"):
        build_migration_plan(source, engine)
    engine.dispose()


def test_source_snapshot_is_integrity_checked_and_preserves_counts(tmp_path):
    source = tmp_path / "source.db"
    _make_source(source)

    snapshot, counts, digest = _snapshot_source(source, tmp_path / "backups")

    assert snapshot.exists()
    assert len(digest) == 64
    assert counts["patients"] == 1
    with sqlite3.connect(snapshot) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    with sqlite3.connect(source) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM patients"
        ).fetchone()[0] == 1


def test_preflight_rejects_unsupported_sqlite_schema_version(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _make_source(source)
    with sqlite3.connect(source) as connection:
        connection.execute("PRAGMA user_version = 1")
    engine = _make_target(target)

    with pytest.raises(DataMigrationError, match="schema version"):
        build_migration_plan(source, engine)
    engine.dispose()
