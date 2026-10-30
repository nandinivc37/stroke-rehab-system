"""
Additive, idempotent schema migrations for data/rehab.db.

Rules:
  * Only ADD columns/tables; never drop, rename or rewrite data.
  * Existing rows keep NULL in new columns: unavailable historical
    measurements are not backfilled or estimated.
  * Safe to rerun: progress is tracked with PRAGMA user_version and
    every column is checked before being added.
  * A verified backup is made before migrating an existing database.

Usage (from the project root):
    python -m src.database.migrations            # backup + migrate
    python -m src.database.migrations --check    # report only
"""

import argparse
import hashlib
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 3

SESSION_COLUMNS = [
    # What was performed, and with which prescription/configuration.
    ("exercise_id", "TEXT"),
    ("affected_side", "TEXT"),
    ("prescription_id", "INTEGER"),
    ("parameters_json", "TEXT"),
    # 1 = generic ExerciseAnalyzer with the exercise catalogue.
    # NULL = recorded by the original implementation (unknown version).
    ("metrics_version", "INTEGER"),
    # Performance metrics.
    ("target_reps", "INTEGER"),
    ("off_plane_reps", "INTEGER"),
    ("trajectory_similarity", "REAL"),
    ("valid_segments", "INTEGER"),
    ("mean_velocity_peaks", "REAL"),
    ("segments_json", "TEXT"),
    # Data quality.
    ("duration_s", "REAL"),
    ("frames_total", "INTEGER"),
    ("frames_with_pose", "INTEGER"),
    ("frames_measured", "INTEGER"),
    ("frame_errors", "INTEGER"),
    ("mean_fps", "REAL"),
    # Experimental, unvalidated posture measurements (JSON).
    ("experimental_posture_json", "TEXT"),
    # Compensation classifier output. Stays NULL until a validated
    # method exists; NULL means "not measured", never "none detected".
    ("compensation_method", "TEXT"),
    ("comp_windows_total", "INTEGER"),
    ("comp_no_compensation", "INTEGER"),
    ("comp_shoulder_elevation", "INTEGER"),
    ("comp_trunk_rotation", "INTEGER"),
    ("comp_lean_forward", "INTEGER"),
    # --- Schema 2 (metrics v2) ---
    # Peak ROM from in-plane movements (3-frame median); NULL when no
    # such movement was measured. max_angle keeps the raw session max.
    ("peak_rom", "REAL"),
    # Mean DTW similarity (0-100) and mean matched-sample error (deg)
    # of the scored segments, reported separately from the movement
    # score stored in trajectory_similarity.
    ("dtw_similarity", "REAL"),
    ("dtw_mean_error_deg", "REAL"),
    # COMPLETE, ANALYSIS_HALTED, CAMERA_LOST, INTERRUPTED_APP_CLOSED.
    # NULL on sessions recorded before schema 2.
    ("session_status", "TEXT"),
    ("interrupted_reps", "INTEGER"),
    # Pipeline configuration (pose model, threading, frame size) and
    # per-stage timing (JSON).
    ("diagnostics_json", "TEXT"),
    # --- Schema 3 ---
    # Rule-based compensation detection summary (JSON; method in
    # compensation_method). Provisional thresholds, not validated, never
    # used for adaptation. NULL = not run (sessions before schema 3).
    ("compensation_json", "TEXT"),
]

RECOMMENDATION_COLUMNS = [
    ("exercise_id", "TEXT"),
    ("affected_side", "TEXT"),
    ("parameters_json", "TEXT"),
    # 'ASSESSMENT' or 'ADAPTATION'. NULL on rows created before this
    # migration (those all came from the assessment workflow).
    ("source", "TEXT"),
    ("adaptation_decision_id", "INTEGER"),
    ("reviewed_by", "TEXT"),
    ("review_note", "TEXT"),
]

ADAPTATION_TABLE = """
CREATE TABLE IF NOT EXISTS adaptation_decisions(
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    exercise_id TEXT NOT NULL,
    affected_side TEXT NOT NULL,
    session_id INTEGER,
    policy_version TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    previous_parameters_json TEXT,
    proposed_parameters_json TEXT,
    previous_recommendation_id INTEGER,
    proposed_recommendation_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
)
"""


# Per-frame diagnostics for investigating measurement errors. Landmark
# numbers only: no images or video are stored.
SESSION_FRAMES_TABLE = """
CREATE TABLE IF NOT EXISTS session_frames(
    session_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    t_s REAL,
    status TEXT NOT NULL,
    angle REAL,
    plane REAL,
    min_visibility REAL,
    reference_frame TEXT,
    PRIMARY KEY(session_id, frame_index),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
)
"""


class MigrationError(RuntimeError):
    pass


# ---------------------------------------------------------
# Backup
# ---------------------------------------------------------

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_row_counts(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name"
            )
        ]
        return {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
    finally:
        conn.close()


def integrity_ok(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def backup_database(db_path, backup_dir):
    """
    Copy the database file and verify the copy (hash, integrity, row
    counts). Refuses to run if SQLite journal/WAL files are present,
    because then the main file alone may not be a consistent snapshot.

    Returns the verified backup path.
    """

    db_path = Path(db_path)
    backup_dir = Path(backup_dir)

    for suffix in ("-journal", "-wal"):
        if Path(str(db_path) + suffix).exists():
            raise MigrationError(
                f"{db_path.name}{suffix} exists: close the application "
                "before backing up."
            )

    if not integrity_ok(db_path):
        raise MigrationError(f"Integrity check failed for {db_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"{db_path.stem}_{stamp}.db"

    shutil.copy2(db_path, backup_path)

    if sha256(backup_path) != sha256(db_path):
        raise MigrationError("Backup hash does not match the original.")
    if not integrity_ok(backup_path):
        raise MigrationError("Backup failed the integrity check.")
    if table_row_counts(backup_path) != table_row_counts(db_path):
        raise MigrationError("Backup row counts differ from the original.")

    return backup_path


# ---------------------------------------------------------
# Migration
# ---------------------------------------------------------

def schema_version(conn):
    return conn.execute("PRAGMA user_version").fetchone()[0]


def existing_columns(conn, table):
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _add_columns(conn, table, columns):
    present = existing_columns(conn, table)
    added = []
    for name, sql_type in columns:
        if name not in present:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN {name} {sql_type}')
            added.append(name)
    return added


def migrate(conn):
    """
    Apply all pending migrations inside one transaction.
    Returns a dict describing what was added (empty when up to date).
    """

    if schema_version(conn) > SCHEMA_VERSION:
        raise MigrationError(
            "Database schema is newer than this application supports."
        )

    for table in ("sessions", "exercise_recommendations"):
        if not existing_columns(conn, table):
            raise MigrationError(f"Required table {table!r} is missing.")

    changes = {}

    conn.execute("BEGIN")
    try:
        changes["sessions"] = _add_columns(conn, "sessions", SESSION_COLUMNS)
        changes["exercise_recommendations"] = _add_columns(
            conn, "exercise_recommendations", RECOMMENDATION_COLUMNS
        )

        had_table = bool(existing_columns(conn, "adaptation_decisions"))
        conn.execute(ADAPTATION_TABLE)
        changes["adaptation_decisions"] = [] if had_table else ["<created>"]

        had_frames = bool(existing_columns(conn, "session_frames"))
        conn.execute(SESSION_FRAMES_TABLE)
        changes["session_frames"] = [] if had_frames else ["<created>"]

        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return {table: cols for table, cols in changes.items() if cols}


def connect_for_migration(db_path):
    # isolation_level=None: transactions are controlled explicitly.
    return sqlite3.connect(db_path, isolation_level=None)


def ensure_schema(db_path, backup_dir):
    """
    Bring a database up to SCHEMA_VERSION, backing it up first if it
    already contains data. Returns (backup_path or None, changes).
    """

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_for_migration(db_path)
    try:
        current = schema_version(conn)
        has_data = any(
            count for count in (
                conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                for t in ("patients", "sessions")
                if existing_columns(conn, t)
            )
        )
    finally:
        conn.close()

    if current >= SCHEMA_VERSION:
        return None, {}

    backup_path = backup_database(db_path, backup_dir) if has_data else None

    conn = connect_for_migration(db_path)
    try:
        changes = migrate(conn)
    finally:
        conn.close()

    return backup_path, changes


def main():
    from src.paths import DB_BACKUP_DIR, DB_PATH

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--backup-dir", type=Path, default=DB_BACKUP_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    version = schema_version(conn)
    conn.close()

    print(f"Database: {args.db}")
    print(f"Schema version: {version} (application: {SCHEMA_VERSION})")

    if args.check:
        return

    backup_path, changes = ensure_schema(args.db, args.backup_dir)

    if backup_path:
        print(f"Verified backup: {backup_path}")
    print(f"Changes: {changes or 'none (already up to date)'}")


if __name__ == "__main__":
    main()
