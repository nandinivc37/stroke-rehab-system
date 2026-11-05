"""Patient application services backed by the SQLAlchemy repository."""

from sqlalchemy.orm import Session

from src.database.repositories import patient_repository


class PatientNotFoundError(LookupError):
    """Raised when a requested patient does not exist."""


def list_patients(session: Session):
    return patient_repository.list_patients(session)


def get_patient(session: Session, patient_id: int):
    patient = patient_repository.get_patient(session, patient_id)
    if patient is None:
        raise PatientNotFoundError(f"Patient {patient_id} was not found.")
    return patient


def create_patient(session: Session, payload):
    name = payload.name.strip()
    if not name:
        raise ValueError("Patient name cannot be blank.")
    return patient_repository.create_patient(
        session,
        name=name,
        age=payload.age,
        gender=payload.gender,
        therapist=payload.therapist,
        notes=payload.notes,
    )


def list_affected_areas(session: Session, patient_id: int):
    get_patient(session, patient_id)
    return patient_repository.list_affected_areas(session, patient_id)


def add_affected_area(session: Session, patient_id: int, payload):
    get_patient(session, patient_id)
    return patient_repository.create_affected_area(
        session,
        patient_id=patient_id,
        side=payload.side,
        body_region=payload.body_region,
        joint=payload.joint,
        notes=payload.notes,
    )
