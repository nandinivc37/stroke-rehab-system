"""Patient and affected-area endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session, get_current_user, require_patient_access, require_roles
from src.schemas.patient import (
    AffectedAreaCreate,
    AffectedAreaRead,
    PatientCreate,
    PatientRead,
)
from src.services import patient_service
from src.services.patient_service import PatientNotFoundError

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("", response_model=list[PatientRead])
def list_patients(
    session: Session = Depends(get_db_session),
    user=Depends(get_current_user),
):
    patients = patient_service.list_patients(session)
    if user["role"] == "admin":
        return patients
    from src.database.repositories import assignment_repository
    allowed = set(assignment_repository.list_assigned_patient_ids(
        session, user["user_id"]
    ))
    return [patient for patient in patients if patient.patient_id in allowed]


@router.post("", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient(
    payload: PatientCreate,
    session: Session = Depends(get_db_session),
    user=Depends(get_current_user),
):
    try:
        patient = patient_service.create_patient(session, payload)
        if user["role"] == "therapist":
            from src.database.repositories import assignment_repository
            assignment_repository.assign_therapist(
                session, patient_id=patient.patient_id,
                therapist_user_id=user["user_id"],
                assigned_by_user_id=user["user_id"],
            )
        return patient
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(
    patient_id: int,
    session: Session = Depends(get_db_session),
    _access=Depends(require_patient_access),
):
    try:
        return patient_service.get_patient(patient_id, session=session)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{patient_id}/affected-areas", response_model=list[AffectedAreaRead])
def list_affected_areas(
    patient_id: int, session: Session = Depends(get_db_session),
    _access=Depends(require_patient_access),
):
    try:
        return patient_service.list_affected_areas(session, patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{patient_id}/affected-areas",
    response_model=AffectedAreaRead,
    status_code=status.HTTP_201_CREATED,
)
def create_affected_area(
    patient_id: int,
    payload: AffectedAreaCreate,
    session: Session = Depends(get_db_session),
    _access=Depends(require_patient_access),
):
    try:
        return patient_service.add_affected_area(session, patient_id, payload)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc



@router.put("/{patient_id}/therapists/{therapist_user_id}", status_code=status.HTTP_200_OK)
def assign_patient_to_therapist(
    patient_id: int,
    therapist_user_id: int,
    session: Session = Depends(get_db_session),
    admin=Depends(require_roles("admin")),
):
    from src.database.models import ApiUser
    from src.database.repositories import assignment_repository

    patient = patient_service.get_patient(patient_id, session=session)
    therapist = session.get(ApiUser, therapist_user_id)
    if therapist is None or therapist.role != "therapist" or not therapist.is_active:
        raise HTTPException(status_code=404, detail="Active therapist was not found.")
    assignment = assignment_repository.assign_therapist(
        session, patient_id=patient.patient_id,
        therapist_user_id=therapist_user_id,
        assigned_by_user_id=admin["user_id"],
    )
    return {
        "patient_id": assignment.patient_id,
        "therapist_user_id": assignment.therapist_user_id,
        "assigned_by_user_id": assignment.assigned_by_user_id,
    }
