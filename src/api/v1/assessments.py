"""Clinical assessment and therapist-reviewed recommendation endpoints."""

from fastapi import APIRouter, HTTPException, status

from src.schemas.assessment import AssessmentCreate, AssessmentRead, RecommendationReview
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
def generate_recommendation(patient_id: int, payload: dict):
    assessment_id = payload.get("assessment_id")
    if not isinstance(assessment_id, int) or assessment_id <= 0:
        raise HTTPException(status_code=422, detail="A positive assessment_id is required.")
    try:
        return assessment_service.generate_recommendation(patient_id, assessment_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except assessment_service.AssessmentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/../recommendations/{recommendation_id}/approve")
def approve_recommendation(recommendation_id: int, payload: RecommendationReview):
    try:
        return assessment_service.review_recommendation(
            recommendation_id, payload, approve=True
        )
    except assessment_service.RecommendationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except assessment_service.RecommendationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/../recommendations/{recommendation_id}/reject")
def reject_recommendation(recommendation_id: int, payload: RecommendationReview):
    try:
        return assessment_service.review_recommendation(
            recommendation_id, payload, approve=False
        )
    except assessment_service.RecommendationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except assessment_service.RecommendationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
