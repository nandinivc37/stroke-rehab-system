"""Patient application services with a compatibility bridge to the legacy data layer."""

from sqlalchemy.orm import Session

from src.database import db
from src.database.repositories import patient_repository


class PatientNotFoundError(LookupError):
    """Raised when a requested patient does not exist."""


def list_patients(session: Session):
    return patient_repository.list_patients(session)


def get_patient(patient_id: int, session: Session | None = None):
    """Use the ORM repository when a request session is supplied.

    Existing application services still call get_patient(patient_id); keep
    that call shape on the legacy SQLite layer until those services migrate.
    """
    patient = (
        patient_repository.get_patient(session, patient_id)
        if session is not None
        else db.get_patient(patient_id)
    )
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
    get_patient(patient_id, session=session)
    return patient_repository.list_affected_areas(session, patient_id)


def add_affected_area(session: Session, patient_id: int, payload):
    get_patient(patient_id, session=session)
    return patient_repository.create_affected_area(
        session,
        patient_id=patient_id,
        side=payload.side,
        body_region=payload.body_region,
        joint=payload.joint,
        notes=payload.notes,
    )
