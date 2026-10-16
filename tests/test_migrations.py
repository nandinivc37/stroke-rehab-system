"""WS4: additive, idempotent migration with verified backup."""

import shutil
import sqlite3

import pytest

from src.database import migrations as m
from src.paths import DB_PATH

# Schema exactly as created by the original application (commit 22e71b3).
LEGACY_SCHEMA = """
CREATE TABLE patients(
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL, age INTEGER, gender TEXT, therapist TEXT, notes TEXT);
CREATE TABLE affected_areas(
    area_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL, side TEXT NOT NULL,
    body_region TEXT NOT NULL, joint TEXT, notes TEXT);
CREATE TABLE sessions(
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL, exercise TEXT, reps INTEGER,
    max_angle REAL, min_angle REAL,
    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE clinical_assessments(
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL, diagnosis TEXT NOT NULL,
    impairment TEXT NOT NULL, baseline_rom REAL, restrictions TEXT,
    clinical_notes TEXT,
    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE exercise_recommendations(
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL, assessment_id INTEGER NOT NULL,
    exercise TEXT NOT NULL, difficulty TEXT NOT NULL, reason TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, approved_at TIMESTAMP);
"""

ORIGINAL_SESSION_COLUMNS = (
    "session_id, patient_id, exercise, reps, max_angle, min_angle, "
    "session_date"
)
ORIGINAL_RECOMMENDATION_COLUMNS = (
    "recommendation_id, patient_id, assessment_id, exercise, difficulty, "
    "reason, status, created_at, approved_at"
)


@pytest.fixture
def legacy_db(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    conn.execute("INSERT INTO patients(name) VALUES ('Synthetic A')")
    conn.executemany(
        "INSERT INTO sessions(patient_id, exercise, reps, max_angle, "
        "min_angle, session_date) VALUES (1, ?, ?, ?, ?, ?)",
        [
            ("Shoulder Flexion", 4, 172.97, 0.09, "2026-10-01 10:00:00"),
            ("Elbow Flexion", 0, None, None, "2026-10-02 10:00:00"),
        ],
    )
    conn.execute(
        "INSERT INTO exercise_recommendations(patient_id, assessment_id, "
        "exercise, difficulty, status) VALUES (1, 1, 'Shoulder Flexion', "
        "'Easy', 'APPROVED')"
    )
    conn.commit()
    conn.close()
    return path


def snapshot(path, table, columns):
    conn = sqlite3.connect(path)
    rows = conn.execute(f"SELECT {columns} FROM {table} ORDER BY 1").fetchall()
    conn.close()
    return rows


def test_backup_is_verified_copy(legacy_db, tmp_path):
    backup = m.backup_database(legacy_db, tmp_path / "backups")
    assert backup.exists()
    assert m.sha256(backup) == m.sha256(legacy_db)
    assert m.table_row_counts(backup) == m.table_row_counts(legacy_db)


def test_backup_refuses_when_journal_present(legacy_db, tmp_path):
    (tmp_path / "legacy.db-journal").write_bytes(b"")
    with pytest.raises(m.MigrationError):
        m.backup_database(legacy_db, tmp_path / "backups")


def test_migration_preserves_rows_and_leaves_new_columns_null(
    legacy_db, tmp_path
):
    sessions_before = snapshot(legacy_db, "sessions", ORIGINAL_SESSION_COLUMNS)
    recs_before = snapshot(
        legacy_db, "exercise_recommendations", ORIGINAL_RECOMMENDATION_COLUMNS
    )

    backup, changes = m.ensure_schema(legacy_db, tmp_path / "backups")

    assert backup is not None and backup.exists()
    assert set(changes["sessions"]) == {c for c, _ in m.SESSION_COLUMNS}
    assert "adaptation_decisions" in changes

    assert snapshot(legacy_db, "sessions", ORIGINAL_SESSION_COLUMNS) == (
        sessions_before
    )
    assert snapshot(
        legacy_db, "exercise_recommendations", ORIGINAL_RECOMMENDATION_COLUMNS
    ) == recs_before

    new_columns = ", ".join(c for c, _ in m.SESSION_COLUMNS)
    for row in snapshot(legacy_db, "sessions", new_columns):
        assert all(value is None for value in row)

    # The backup still has the original schema.
    conn = sqlite3.connect(backup)
    assert m.schema_version(conn) == 0
    assert "trajectory_similarity" not in m.existing_columns(conn, "sessions")
    conn.close()


def test_migration_is_idempotent(legacy_db, tmp_path):
    m.ensure_schema(legacy_db, tmp_path / "backups")
    hash_after_first = m.sha256(legacy_db)

    backup, changes = m.ensure_schema(legacy_db, tmp_path / "backups")
    assert backup is None and changes == {}

    # Running migrate() directly again also changes nothing.
    conn = m.connect_for_migration(legacy_db)
    assert m.migrate(conn) == {}
    conn.close()

    conn = sqlite3.connect(legacy_db)
    assert m.schema_version(conn) == m.SCHEMA_VERSION
    conn.close()
    assert len(list((tmp_path / "backups").iterdir())) == 1
    assert hash_after_first  # file still readable


def test_partially_migrated_database_is_completed(legacy_db, tmp_path):
    conn = sqlite3.connect(legacy_db)
    conn.execute("ALTER TABLE sessions ADD COLUMN trajectory_similarity REAL")
    conn.commit()
    conn.close()

    _, changes = m.ensure_schema(legacy_db, tmp_path / "backups")
    assert "trajectory_similarity" not in changes["sessions"]

    conn = sqlite3.connect(legacy_db)
    columns = m.existing_columns(conn, "sessions")
    conn.close()
    assert {c for c, _ in m.SESSION_COLUMNS} <= columns


def test_newer_schema_is_rejected(legacy_db):
    conn = m.connect_for_migration(legacy_db)
    conn.execute(f"PRAGMA user_version = {m.SCHEMA_VERSION + 1}")
    with pytest.raises(m.MigrationError):
        m.migrate(conn)
    conn.close()


def test_empty_new_database_needs_no_backup(temp_db):
    conn = sqlite3.connect(temp_db)
    assert m.schema_version(conn) == m.SCHEMA_VERSION
    assert "adaptation_decisions" in {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    conn.close()
    assert not (temp_db.parent / "backups").exists()


@pytest.mark.skipif(not DB_PATH.exists(), reason="no local rehab.db")
def test_migration_on_copy_of_real_database(tmp_path):
    """Dry run on a COPY of data/rehab.db; the real file is not modified."""

    original_hash = m.sha256(DB_PATH)
    copy = tmp_path / "rehab_copy.db"
    shutil.copy2(DB_PATH, copy)

    conn = sqlite3.connect(copy)
    already = m.schema_version(conn) >= m.SCHEMA_VERSION
    conn.close()

    before = snapshot(copy, "sessions", ORIGINAL_SESSION_COLUMNS)
    counts_before = m.table_row_counts(copy)

    m.ensure_schema(copy, tmp_path / "backups")

    assert snapshot(copy, "sessions", ORIGINAL_SESSION_COLUMNS) == before
    counts_after = m.table_row_counts(copy)
    for table, count in counts_before.items():
        assert counts_after[table] == count

    if not already:
        new_columns = ", ".join(c for c, _ in m.SESSION_COLUMNS)
        for row in snapshot(copy, "sessions", new_columns):
            assert all(value is None for value in row)

    assert m.sha256(DB_PATH) == original_hash
