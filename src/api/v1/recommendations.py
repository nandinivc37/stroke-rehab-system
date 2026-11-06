"""Therapist review endpoints for generated exercise recommendations."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session, require_recommendation_access
from src.schemas.assessment import RecommendationReview
from src.services import assessment_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/{recommendation_id}/approve")
def approve_recommendation(recommendation_id: int, payload: RecommendationReview, session: Session = Depends(get_db_session), reviewer=Depends(require_recommendation_access)):
    try:
        payload = payload.model_copy(update={"reviewer": reviewer["username"]})
        return assessment_service.review_recommendation(
            session, recommendation_id, payload, approve=True
        )
    except assessment_service.RecommendationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except assessment_service.RecommendationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{recommendation_id}/reject")
def reject_recommendation(recommendation_id: int, payload: RecommendationReview, session: Session = Depends(get_db_session), reviewer=Depends(require_recommendation_access)):
    try:
        payload = payload.model_copy(update={"reviewer": reviewer["username"]})
        return assessment_service.review_recommendation(
            session, recommendation_id, payload, approve=False
        )
    except assessment_service.RecommendationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except assessment_service.RecommendationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
