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


if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully!")