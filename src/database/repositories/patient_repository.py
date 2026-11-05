"""SQLAlchemy repository for patient and affected-area records."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.db import normalize_body_region
from src.database.models import AffectedArea, Patient


def list_patients(session: Session):
    return list(session.scalars(select(Patient).order_by(Patient.patient_id)))


def get_patient(session: Session, patient_id: int):
    return session.get(Patient, patient_id)


def create_patient(session: Session, *, name, age=None, gender=None, therapist=None, notes=None):
    patient = Patient(
        name=name,
        age=age,
        gender=gender,
        therapist=therapist,
        notes=notes,
    )
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return patient


def list_affected_areas(session: Session, patient_id: int):
    areas = session.scalars(
        select(AffectedArea)
        .where(AffectedArea.patient_id == patient_id)
        .order_by(AffectedArea.area_id)
    )
    for area in areas:
        area.body_region = normalize_body_region(area.body_region, area.joint)
    return list(areas)


def create_affected_area(
    session: Session, *, patient_id: int, side: str, body_region: str, joint=None, notes=None
):
    area = AffectedArea(
        patient_id=patient_id,
        side=side,
        body_region=normalize_body_region(body_region, joint),
        joint=joint,
        notes=notes,
    )
    session.add(area)
    session.commit()
    session.refresh(area)
    return area
