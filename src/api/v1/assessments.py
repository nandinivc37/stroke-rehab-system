"""Clinical assessment and therapist-reviewed recommendation endpoints."""

from fastapi import APIRouter, HTTPException, status

from src.schemas.assessment import AssessmentCreate, AssessmentRead, RecommendationGenerate
from src.services import assessment_service
from src.services.patient_service import PatientNotFoundError

router = APIRouter(prefix="/patients", tags=["clinical assessments"])


@router.post(
    "/{patient_id}/assessments",
    response_model=AssessmentRead,
    status_code=status.HTTP_201_CREATED,
)
def create_assessment(patient_id: int, payload: AssessmentCreate):
    try:
        return assessment_service.create_assessment(patient_id, payload)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{patient_id}/assessments/current", response_model=AssessmentRead)
def get_current_assessment(patient_id: int):
    try:
        return assessment_service.get_latest_assessment(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except assessment_service.AssessmentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{patient_id}/recommendations/generate")
def generate_recommendation(patient_id: int, payload: RecommendationGenerate):
    assessment_id = payload.assessment_id
    try:
        return assessment_service.generate_recommendation(patient_id, assessment_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except assessment_service.AssessmentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

