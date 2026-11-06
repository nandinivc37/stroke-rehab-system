"""Repository for explicit therapist-to-patient access grants."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import PatientTherapistAssignment


def assign_therapist(
    session: Session, *, patient_id: int, therapist_user_id: int, assigned_by_user_id: int
):
    existing = session.scalars(
        select(PatientTherapistAssignment).where(
            PatientTherapistAssignment.patient_id == patient_id,
            PatientTherapistAssignment.therapist_user_id == therapist_user_id,
        )
    ).first()
    if existing is not None:
        return existing
    row = PatientTherapistAssignment(
        patient_id=patient_id,
        therapist_user_id=therapist_user_id,
        assigned_by_user_id=assigned_by_user_id,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def therapist_has_patient(session: Session, *, patient_id: int, therapist_user_id: int) -> bool:
    return session.scalars(
        select(PatientTherapistAssignment.assignment_id).where(
            PatientTherapistAssignment.patient_id == patient_id,
            PatientTherapistAssignment.therapist_user_id == therapist_user_id,
        ).limit(1)
    ).first() is not None


def list_assigned_patient_ids(session: Session, therapist_user_id: int) -> list[int]:
    return list(session.scalars(
        select(PatientTherapistAssignment.patient_id)
        .where(PatientTherapistAssignment.therapist_user_id == therapist_user_id)
        .order_by(PatientTherapistAssignment.patient_id)
    ))
