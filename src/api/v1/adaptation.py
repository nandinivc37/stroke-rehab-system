"""Adaptive decision-support and therapist-review API endpoints."""

from fastapi import APIRouter, HTTPException

from src.adaptation.workflow import AdaptationError
from src.schemas.adaptation import AdaptationEvaluate, AdaptationReview
from src.services import adaptation_service
from src.services.patient_service import PatientNotFoundError

router = APIRouter(tags=["adaptive decision support"])


@router.post("/patients/{patient_id}/sessions/{session_id}/adaptation/evaluate")
def evaluate_session(patient_id: int, session_id: int):
    try:
        return adaptation_service.evaluate_session(patient_id, session_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AdaptationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/patients/{patient_id}/adaptation/pending")
def pending_proposals(patient_id: int):
    try:
        return adaptation_service.pending_proposals(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/patients/{patient_id}/adaptation/history")
def decision_history(patient_id: int):
    try:
        return adaptation_service.decision_history(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/adaptation/{proposal_id}/approve")
def approve_proposal(proposal_id: int, payload: AdaptationReview):
    try:
        return adaptation_service.review_proposal(
            proposal_id, True, payload.reviewer, payload.note
        )
    except AdaptationError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post("/adaptation/{proposal_id}/reject")
def reject_proposal(proposal_id: int, payload: AdaptationReview):
    try:
        return adaptation_service.review_proposal(
            proposal_id, False, payload.reviewer, payload.note
        )
    except AdaptationError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
