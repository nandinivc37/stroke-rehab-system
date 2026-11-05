"""Patient and affected-area endpoints."""

from fastapi import APIRouter, HTTPException, status

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
def list_patients():
    return patient_service.list_patients()


@router.post("", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate):
    try:
        return patient_service.create_patient(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: int):
    try:
        return patient_service.get_patient(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{patient_id}/affected-areas", response_model=list[AffectedAreaRead])
def list_affected_areas(patient_id: int):
    try:
        return patient_service.list_affected_areas(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{patient_id}/affected-areas",
    response_model=AffectedAreaRead,
    status_code=status.HTTP_201_CREATED,
)
def create_affected_area(patient_id: int, payload: AffectedAreaCreate):
    try:
        return patient_service.add_affected_area(patient_id, payload)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
