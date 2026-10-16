import json
import sqlite3
from pathlib import Path

from src.database.migrations import ensure_schema
from src.paths import DB_PATH

# Database path (module-level so tests can point it at a temporary copy)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    # -------------------------
    # Patient table
    # -------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        therapist TEXT,
        notes TEXT
    )
    """)

    # -------------------------
    # Affected areas table
    # -------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS affected_areas(
        area_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        side TEXT NOT NULL,
        body_region TEXT NOT NULL,
        joint TEXT,
        notes TEXT,
        FOREIGN KEY(patient_id)
            REFERENCES patients(patient_id)
    )
    """)

    # -------------------------
    # Rehabilitation sessions
    # -------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions(
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        exercise TEXT,
        reps INTEGER,
        max_angle REAL,
        min_angle REAL,
        session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id)
            REFERENCES patients(patient_id)
    )
    """)

    # -------------------------
    # Clinical assessments
    # -------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clinical_assessments(
        assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        diagnosis TEXT NOT NULL,
        impairment TEXT NOT NULL,
        baseline_rom REAL,
        restrictions TEXT,
        clinical_notes TEXT,
        assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id)
            REFERENCES patients(patient_id)
    )
    """)

    # -------------------------
    # Exercise recommendations
    # -------------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS exercise_recommendations(
        recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        assessment_id INTEGER NOT NULL,
        exercise TEXT NOT NULL,
        difficulty TEXT NOT NULL,
        reason TEXT,
        status TEXT NOT NULL DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        approved_at TIMESTAMP,
        FOREIGN KEY(patient_id)
            REFERENCES patients(patient_id),
        FOREIGN KEY(assessment_id)
            REFERENCES clinical_assessments(assessment_id)
    )
    """)

    conn.commit()
    conn.close()

    # Additive schema upgrades (verified backup first if data exists).
    ensure_schema(DB_PATH, Path(DB_PATH).parent / "backups")


def add_patient(name, age, gender, therapist, notes):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO patients(
            name,
            age,
            gender,
            therapist,
            notes
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        name,
        age,
        gender,
        therapist,
        notes
    ))

    conn.commit()

    patient_id = cursor.lastrowid

    conn.close()

    return patient_id


# Legacy affected-area values. "Arm" was stored for at least one patient
# before the registration form's current body regions (Upper Limb, Lower
# Limb, Trunk, Full Body); it never appeared in any committed version of
# the form. It is read as "Upper Limb" only when the joint is an
# upper-limb joint. Stored rows are not rewritten.
LEGACY_BODY_REGIONS = {"arm": "Upper Limb"}
UPPER_LIMB_JOINTS = {"Shoulder", "Elbow", "Wrist"}


def normalize_body_region(body_region, joint):
    """Current body-region name for a (possibly legacy) stored value."""

    if body_region is None:
        return None
    mapped = LEGACY_BODY_REGIONS.get(body_region.strip().casefold())
    if mapped == "Upper Limb" and joint in UPPER_LIMB_JOINTS:
        return mapped
    return body_region


def add_affected_area(
    patient_id,
    side,
    body_region,
    joint,
    notes
):
    body_region = normalize_body_region(body_region, joint)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO affected_areas(
            patient_id,
            side,
            body_region,
            joint,
            notes
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        patient_id,
        side,
        body_region,
        joint,
        notes
    ))

    conn.commit()
    conn.close()


def get_all_patients():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM patients
    """)

    patients = cursor.fetchall()

    conn.close()

    return patients


def get_affected_areas(patient_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            area_id,
            side,
            body_region,
            joint,
            notes
        FROM affected_areas
        WHERE patient_id = ?
    """, (patient_id,))

    areas = [
        (area_id, side, normalize_body_region(body_region, joint), joint,
         notes)
        for area_id, side, body_region, joint, notes in cursor.fetchall()
    ]

    conn.close()

    return areas


def save_session(
    patient_id,
    exercise,
    reps,
    max_angle,
    min_angle
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sessions(
            patient_id,
            exercise,
            reps,
            max_angle,
            min_angle
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        patient_id,
        exercise,
        reps,
        max_angle,
        min_angle
    ))

    conn.commit()
    conn.close()

def save_clinical_assessment(
    patient_id,
    diagnosis,
    impairment,
    baseline_rom,
    restrictions,
    clinical_notes
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO clinical_assessments(
            patient_id,
            diagnosis,
            impairment,
            baseline_rom,
            restrictions,
            clinical_notes
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        diagnosis,
        impairment,
        baseline_rom,
        restrictions,
        clinical_notes
    ))

    conn.commit()
    assessment_id = cursor.lastrowid
    conn.close()

    return assessment_id

def get_latest_assessment(patient_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            assessment_id,
            diagnosis,
            impairment,
            baseline_rom,
            restrictions,
            clinical_notes,
            assessment_date
        FROM clinical_assessments
        WHERE patient_id = ?
        ORDER BY assessment_date DESC
        LIMIT 1
    """, (patient_id,))

    assessment = cursor.fetchone()
    conn.close()

    return assessment

def save_recommendation(
    patient_id,
    assessment_id,
    exercise,
    difficulty,
    reason
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO exercise_recommendations(
            patient_id,
            assessment_id,
            exercise,
            difficulty,
            reason,
            status
        )
        VALUES (?, ?, ?, ?, ?, 'PENDING')
    """, (
        patient_id,
        assessment_id,
        exercise,
        difficulty,
        reason
    ))

    conn.commit()
    recommendation_id = cursor.lastrowid
    conn.close()

    return recommendation_id


def update_recommendation_status(
    recommendation_id,
    status
):
    conn = get_connection()
    cursor = conn.cursor()

    if status == "APPROVED":
        cursor.execute("""
            UPDATE exercise_recommendations
            SET status = ?,
                approved_at = CURRENT_TIMESTAMP
            WHERE recommendation_id = ?
        """, (status, recommendation_id))

    else:
        cursor.execute("""
            UPDATE exercise_recommendations
            SET status = ?
            WHERE recommendation_id = ?
        """, (status, recommendation_id))

    conn.commit()
    conn.close()


def get_latest_recommendation(patient_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            recommendation_id,
            assessment_id,
            exercise,
            difficulty,
            reason,
            status,
            created_at,
            approved_at
        FROM exercise_recommendations
        WHERE patient_id = ?
        ORDER BY recommendation_id DESC
        LIMIT 1
    """, (patient_id,))

    recommendation = cursor.fetchone()

    conn.close()

    return recommendation


def get_latest_approved_exercise(patient_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            recommendation_id,
            exercise,
            difficulty,
            approved_at
        FROM exercise_recommendations
        WHERE patient_id = ?
          AND status = 'APPROVED'
        ORDER BY recommendation_id DESC
        LIMIT 1
    """, (patient_id,))

    exercise = cursor.fetchone()

    conn.close()

    return exercise


if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully!")

# =========================================================
# Prescriptions, session persistence and history
# (schema version 1; see src/database/migrations.py)
# =========================================================


# Metrics produced by src/exercises/analyzer.py with the exercise
# catalogue. Sessions recorded before this have metrics_version NULL.
METRICS_VERSION = 1


def _dict_rows(cursor):
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _json_or_none(value):
    return None if value is None else json.dumps(value, sort_keys=True)


def get_active_prescription(patient_id):
    """
    Latest APPROVED exercise prescription for a patient, as a dict with
    'parameters' decoded (empty dict when none were stored).
    """

    conn = get_connection()
    cursor = conn.execute("""
        SELECT *
        FROM exercise_recommendations
        WHERE patient_id = ?
          AND status = 'APPROVED'
        ORDER BY recommendation_id DESC
        LIMIT 1
    """, (patient_id,))
    rows = _dict_rows(cursor)
    conn.close()

    if not rows:
        return None

    row = rows[0]
    row["parameters"] = (
        json.loads(row["parameters_json"]) if row["parameters_json"] else {}
    )
    return row


def set_recommendation_prescription(
    recommendation_id,
    exercise,
    exercise_id,
    affected_side,
    parameters,
    source,
):
    """Record which exercise/parameters a PENDING recommendation assigns."""

    conn = get_connection()
    conn.execute("""
        UPDATE exercise_recommendations
        SET exercise = ?,
            exercise_id = ?,
            affected_side = ?,
            parameters_json = ?,
            source = ?
        WHERE recommendation_id = ?
          AND status = 'PENDING'
    """, (
        exercise,
        exercise_id,
        affected_side,
        _json_or_none(parameters),
        source,
        recommendation_id,
    ))
    conn.commit()
    conn.close()


def save_session_summary(
    patient_id,
    exercise_name,
    stats,
    prescription_id=None,
):
    """
    Persist a session summary produced by ExerciseAnalyzer (plus the
    widget's experimental posture summary). Values that were not
    measured are stored as NULL. Compensation-classifier columns are
    left NULL: no validated classifier is in use.

    Returns the new session_id.
    """

    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO sessions(
            patient_id, exercise, reps, max_angle, min_angle,
            exercise_id, affected_side, prescription_id, parameters_json,
            metrics_version, target_reps, off_plane_reps,
            trajectory_similarity, valid_segments, mean_velocity_peaks,
            segments_json, duration_s, frames_total, frames_with_pose,
            frames_measured, frame_errors, mean_fps,
            experimental_posture_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?)
    """, (
        patient_id,
        exercise_name,
        stats["reps"],
        stats.get("max_angle"),
        stats.get("min_angle"),
        stats.get("exercise_id"),
        stats.get("side"),
        prescription_id,
        _json_or_none(stats.get("parameters")),
        METRICS_VERSION,
        stats.get("target_reps"),
        stats.get("off_plane_reps"),
        stats.get("trajectory_similarity"),
        stats.get("valid_segments"),
        stats.get("mean_velocity_peaks"),
        _json_or_none(stats.get("segments")),
        stats.get("duration_s"),
        stats.get("frames_total"),
        stats.get("frames_with_pose"),
        stats.get("frames_measured"),
        stats.get("frame_errors"),
        stats.get("mean_fps"),
        _json_or_none(stats.get("experimental_posture")),
    ))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id


def get_sessions(patient_id, exercise_id=None, affected_side=None):
    """
    All sessions for a patient, oldest first, as dicts. Historical
    sessions (metrics_version NULL) are included; callers must not mix
    them with current-version metrics.
    """

    query = "SELECT * FROM sessions WHERE patient_id = ?"
    params = [patient_id]

    if exercise_id is not None:
        query += " AND exercise_id = ?"
        params.append(exercise_id)

    if affected_side is not None:
        query += " AND affected_side = ?"
        params.append(affected_side)

    query += " ORDER BY session_date, session_id"

    conn = get_connection()
    rows = _dict_rows(conn.execute(query, params))
    conn.close()
    return rows


def get_patient(patient_id):
    conn = get_connection()
    rows = _dict_rows(conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
    ))
    conn.close()
    return rows[0] if rows else None


def get_dashboard_counts():
    """Counts shown on the dashboard, computed from stored data only."""

    conn = get_connection()
    try:
        def scalar(query):
            return conn.execute(query).fetchone()[0]

        return {
            "patients": scalar("SELECT COUNT(*) FROM patients"),
            "sessions": scalar("SELECT COUNT(*) FROM sessions"),
            "sessions_last_7_days": scalar(
                "SELECT COUNT(*) FROM sessions "
                "WHERE session_date >= datetime('now', '-7 days')"
            ),
            "pending_adaptations": scalar(
                "SELECT COUNT(*) FROM exercise_recommendations "
                "WHERE source = 'ADAPTATION' AND status = 'PENDING'"
            ),
            "last_session": scalar("SELECT MAX(session_date) FROM sessions"),
        }
    finally:
        conn.close()


# ---------------------------------------------------------
# Patient editing and deletion
# ---------------------------------------------------------

class PatientRecordError(RuntimeError):
    pass


# Tables holding a patient's records, children first (deletion order).
# Foreign keys are declared but not enforced by SQLite here, so every
# table is handled explicitly to avoid orphaned rows.
PATIENT_TABLES = (
    "adaptation_decisions",
    "sessions",
    "exercise_recommendations",
    "clinical_assessments",
    "affected_areas",
)


def get_current_assessment(patient_id):
    """The patient's latest clinical assessment as a dict, or None."""

    conn = get_connection()
    rows = _dict_rows(conn.execute("""
        SELECT * FROM clinical_assessments
        WHERE patient_id = ?
        ORDER BY assessment_date DESC, assessment_id DESC
        LIMIT 1
    """, (patient_id,)))
    conn.close()
    return rows[0] if rows else None


def update_assessment_diagnosis(patient_id, assessment_id, diagnosis):
    """
    Change the diagnosis of one clinical assessment of one patient.
    Nothing else changes: sessions, prescriptions and their exercise
    settings keep their values (a new recommendation needs a new
    assessment).
    """

    diagnosis = (diagnosis or "").strip()
    if not diagnosis:
        raise PatientRecordError("A diagnosis is required.")

    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute("""
                UPDATE clinical_assessments
                SET diagnosis = ?
                WHERE assessment_id = ? AND patient_id = ?
            """, (diagnosis, assessment_id, patient_id))
            if cursor.rowcount != 1:
                raise PatientRecordError(
                    "That assessment does not belong to this patient."
                )
    finally:
        conn.close()


def get_patient_record_counts(patient_id):
    """Number of rows per table that belong to the patient."""

    conn = get_connection()
    try:
        return {
            table: conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()[0]
            for table in PATIENT_TABLES
        }
    finally:
        conn.close()


def _cross_patient_references(conn, patient_id):
    """Rows of OTHER patients that point at this patient's records."""

    checks = [
        ("exercise_recommendations", "assessment_id",
         "clinical_assessments", "assessment_id"),
        ("sessions", "prescription_id",
         "exercise_recommendations", "recommendation_id"),
        ("adaptation_decisions", "session_id", "sessions", "session_id"),
        ("adaptation_decisions", "previous_recommendation_id",
         "exercise_recommendations", "recommendation_id"),
        ("adaptation_decisions", "proposed_recommendation_id",
         "exercise_recommendations", "recommendation_id"),
        ("exercise_recommendations", "adaptation_decision_id",
         "adaptation_decisions", "decision_id"),
    ]
    found = []
    for table, column, target, key in checks:
        count = conn.execute(f"""
            SELECT COUNT(*) FROM {table}
            WHERE patient_id != ?
              AND {column} IN (
                  SELECT {key} FROM {target} WHERE patient_id = ?)
        """, (patient_id, patient_id)).fetchone()[0]
        if count:
            found.append(f"{count} {table} row(s) via {column}")
    return found


def delete_patient(patient_id, backup_dir=None):
    """
    Permanently delete a patient and every record that belongs to them
    (affected areas, assessments, prescriptions, sessions, adaptation
    decisions) in one transaction, after a verified database backup.

    Refuses if another patient's records reference this patient's
    records. Returns {"deleted": counts per table, "backup": path}.
    """

    from src.database.migrations import backup_database

    conn = get_connection()
    try:
        if conn.execute("SELECT 1 FROM patients WHERE patient_id = ?",
                        (patient_id,)).fetchone() is None:
            raise PatientRecordError("Patient not found.")
        shared = _cross_patient_references(conn, patient_id)
    finally:
        conn.close()

    if shared:
        raise PatientRecordError(
            "Not deleted: other patients' records reference this "
            "patient's records (" + "; ".join(shared) + ")."
        )

    backup = backup_database(
        DB_PATH, backup_dir or Path(DB_PATH).parent / "backups"
    )

    counts = get_patient_record_counts(patient_id)

    conn = get_connection()
    try:
        with conn:
            for table in PATIENT_TABLES:
                conn.execute(f"DELETE FROM {table} WHERE patient_id = ?",
                             (patient_id,))
            conn.execute("DELETE FROM patients WHERE patient_id = ?",
                         (patient_id,))
            remaining = sum(
                conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE patient_id = ?",
                    (patient_id,),
                ).fetchone()[0]
                for table in PATIENT_TABLES + ("patients",)
            )
            if remaining:
                raise PatientRecordError(
                    "Deletion incomplete; nothing was deleted."
                )
    finally:
        conn.close()

    return {"deleted": counts, "backup": backup}
