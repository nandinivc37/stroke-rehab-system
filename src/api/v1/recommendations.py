"""Therapist review endpoints for generated exercise recommendations."""

from fastapi import APIRouter, HTTPException

from src.schemas.assessment import RecommendationReview
from src.services import assessment_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/{recommendation_id}/approve")
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


@router.post("/{recommendation_id}/reject")
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
