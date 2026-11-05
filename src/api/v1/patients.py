"""Patient and affected-area endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
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
def list_patients(session: Session = Depends(get_db_session)):
    return patient_service.list_patients(session)


@router.post("", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate, session: Session = Depends(get_db_session)):
    try:
        return patient_service.create_patient(session, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: int, session: Session = Depends(get_db_session)):
    try:
        return patient_service.get_patient(session, patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{patient_id}/affected-areas", response_model=list[AffectedAreaRead])
def list_affected_areas(patient_id: int, session: Session = Depends(get_db_session)):
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
):
    try:
        return patient_service.add_affected_area(session, patient_id, payload)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
