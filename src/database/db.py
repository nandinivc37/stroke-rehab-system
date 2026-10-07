import sqlite3
from pathlib import Path

# Database path
DB_PATH = Path("data/rehab.db")
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


def add_affected_area(
    patient_id,
    side,
    body_region,
    joint,
    notes
):
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

    areas = cursor.fetchall()

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