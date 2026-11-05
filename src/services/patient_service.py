"""Patient application services backed by the existing SQLite data layer."""

from src.database import db


class PatientNotFoundError(LookupError):
    """Raised when a requested patient does not exist."""


def list_patients():
    rows = db.get_all_patients()
    return [
        {
            "patient_id": row[0],
            "name": row[1],
            "age": row[2],
            "gender": row[3],
            "therapist": row[4],
            "notes": row[5],
        }
        for row in rows
    ]


def get_patient(patient_id: int):
    patient = db.get_patient(patient_id)
    if patient is None:
        raise PatientNotFoundError(f"Patient {patient_id} was not found.")
    return patient


def create_patient(payload):
    name = payload.name.strip()
    if not name:
        raise ValueError("Patient name cannot be blank.")
    patient_id = db.add_patient(
        name=name,
        age=payload.age,
        gender=payload.gender,
        therapist=payload.therapist,
        notes=payload.notes,
    )
    return get_patient(patient_id)


def list_affected_areas(patient_id: int):
    get_patient(patient_id)
    return [
        {
            "area_id": row[0],
            "side": row[1],
            "body_region": row[2],
            "joint": row[3],
            "notes": row[4],
        }
        for row in db.get_affected_areas(patient_id)
    ]


def add_affected_area(patient_id: int, payload):
    get_patient(patient_id)
    db.add_affected_area(
        patient_id=patient_id,
        side=payload.side,
        body_region=payload.body_region,
        joint=payload.joint,
        notes=payload.notes,
    )
    return list_affected_areas(patient_id)[-1]
