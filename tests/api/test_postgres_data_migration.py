"""End-to-end SQLite to PostgreSQL migration test (CI PostgreSQL service)."""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from src.database.migrate_sqlite_to_postgres import apply_migration, verify_migration
from src.database.models import Base


@pytest.mark.skipif(
    not os.environ.get("NARS_TEST_POSTGRES_URL"),
    reason="requires the disposable PostgreSQL service configured in CI",
)
def test_full_synthetic_sqlite_history_migrates_to_postgres(tmp_path, monkeypatch):
    source_path = tmp_path / "nars-synthetic.db"
    backup_dir = tmp_path / "verified-snapshots"
    target_url = os.environ["NARS_TEST_POSTGRES_URL"]

    source_engine = create_engine("sqlite:///" + source_path.as_posix())
    Base.metadata.create_all(source_engine)
    with source_engine.connect() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
        connection.commit()
    with source_engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO patients (patient_id, name, age, therapist) "
            "VALUES (101, 'Synthetic NARS Patient', 62, 'Synthetic Therapist')"
        ))
        connection.execute(text(
            "INSERT INTO affected_areas (area_id, patient_id, side, body_region, joint) "
            "VALUES (201, 101, 'left', 'upper_limb', 'shoulder')"
        ))
        connection.execute(text(
            "INSERT INTO clinical_assessments "
            "(assessment_id, patient_id, diagnosis, impairment, baseline_rom, assessment_date) "
            "VALUES (301, 101, 'stroke', 'reduced_rom', 42.5, '2026-10-01 09:30:00')"
        ))
        connection.execute(text(
            "INSERT INTO exercise_recommendations "
            "(recommendation_id, patient_id, assessment_id, exercise, difficulty, status, created_at) "
            "VALUES (401, 101, 301, 'shoulder_flexion', 'easy', 'APPROVED', '2026-10-01 09:35:00')"
        ))
        connection.execute(text(
            "INSERT INTO sessions "
            "(session_id, patient_id, exercise, reps, peak_rom, dtw_similarity, session_date, session_status) "
            "VALUES (501, 101, 'shoulder_flexion', 5, 58.25, 0.91, '2026-10-02 10:00:00', 'COMPLETED')"
        ))
        connection.execute(text(
            "INSERT INTO session_frames (session_id, frame_index, t_s, status, angle) "
            "VALUES (501, 0, 0.0, 'measured', 12.5), (501, 1, 0.1, 'measured', 13.5)"
        ))
        connection.execute(text(
            "INSERT INTO adaptation_decisions "
            "(decision_id, patient_id, exercise_id, affected_side, session_id, policy_version, decision, rationale, inputs_json) "
            "VALUES (601, 101, 'shoulder_flexion', 'left', 501, 'test-v1', 'maintain', 'synthetic rationale', '{}')"
        ))
        connection.execute(text(
            "INSERT INTO api_users (user_id, username, password_hash, role, is_active) "
            "VALUES (7, 'synthetic-therapist', 'test-hash-not-a-real-credential', 'therapist', 1)"
        ))
        connection.execute(text("PRAGMA user_version = 3"))
    source_engine.dispose()

    monkeypatch.setenv("NARS_DATABASE_URL", target_url)
    command.upgrade(Config("alembic.ini"), "head")
    target_engine = create_engine(target_url)
    try:
        result = apply_migration(source_path, target_engine, backup_dir)
        assert result["verification"]["matches"] is True
        assert result["verification"]["source_counts"] == result["verification"]["target_counts"]
        assert result["snapshot_sha256"]
        assert Path(result["snapshot_path"]).is_file()

        verification = verify_migration(source_path, target_engine)
        assert verification["matches"] is True

        with target_engine.begin() as connection:
            patient = connection.execute(text(
                "SELECT name, age FROM patients WHERE patient_id = 101"
            )).one()
            session = connection.execute(text(
                "SELECT reps, peak_rom, dtw_similarity FROM sessions WHERE session_id = 501"
            )).one()
            frames = connection.execute(text(
                "SELECT COUNT(*) FROM session_frames WHERE session_id = 501"
            )).scalar_one()
            decision = connection.execute(text(
                "SELECT decision, rationale FROM adaptation_decisions WHERE decision_id = 601"
            )).one()
            account = connection.execute(text(
                "SELECT username, role FROM api_users WHERE user_id = 7"
            )).one()
            assert patient == ("Synthetic NARS Patient", 62)
            assert session == (5, 58.25, 0.91)
            assert frames == 2
            assert decision == ("maintain", "synthetic rationale")
            assert account == ("synthetic-therapist", "therapist")
            inserted_id = connection.execute(text(
                "INSERT INTO patients (name) VALUES ('Sequence Check') RETURNING patient_id"
            )).scalar_one()
            assert inserted_id == 102

    finally:
        target_engine.dispose()
