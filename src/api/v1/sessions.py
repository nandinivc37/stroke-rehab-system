"""Landmark-streaming rehabilitation session endpoints."""

from fastapi import APIRouter, HTTPException, status

from src.schemas.session import (
    FrameAnalysisRead,
    PoseFrameCreate,
    SessionCompleteRead,
    SessionStart,
    SessionStarted,
)
from src.services import session_service
from src.services.patient_service import PatientNotFoundError

router = APIRouter(tags=["rehabilitation sessions"])


@router.post(
    "/sessions",
    response_model=SessionStarted,
    status_code=status.HTTP_201_CREATED,
)
def start_session(payload: SessionStart):
    try:
        return session_service.session_manager.start(payload.patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except session_service.SessionValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/sessions/{session_id}/frames",
    response_model=FrameAnalysisRead,
)
def submit_pose_frame(session_id: str, payload: PoseFrameCreate):
    try:
        return session_service.session_manager.add_frame(session_id, payload)
    except session_service.SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except session_service.SessionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/sessions/{session_id}/complete",
    response_model=SessionCompleteRead,
)
def complete_session(session_id: str):
    try:
        return session_service.session_manager.complete(session_id)
    except session_service.SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except session_service.SessionValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
def cancel_session(session_id: str):
    try:
        return session_service.session_manager.cancel(session_id)
    except session_service.SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
