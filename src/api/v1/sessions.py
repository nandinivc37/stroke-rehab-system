"""Landmark-streaming rehabilitation session endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session, require_patient_access, require_session_access
from src.services.auth_service import AuthenticationError, decode_token
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
stream_router = APIRouter(tags=["rehabilitation streaming"])


@router.post(
    "/sessions",
    response_model=SessionStarted,
    status_code=status.HTTP_201_CREATED,
)
def start_session(
    payload: SessionStart,
    database_session: Session = Depends(get_db_session),
    _access=Depends(require_patient_access),
):
    try:
        return session_service.session_manager.start(
            payload.patient_id, database_session=database_session
        )
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except session_service.SessionValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/sessions/{session_id}/frames",
    response_model=FrameAnalysisRead,
)
def submit_pose_frame(
    session_id: str, payload: PoseFrameCreate,
    _access=Depends(require_session_access),
):
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
def complete_session(
    session_id: str,
    database_session: Session = Depends(get_db_session),
    _access=Depends(require_session_access),
):
    try:
        return session_service.session_manager.complete(session_id, database_session)
    except session_service.SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except session_service.SessionValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
def cancel_session(
    session_id: str, database_session: Session = Depends(get_db_session),
    _access=Depends(require_session_access),
):
    try:
        return session_service.session_manager.cancel(
            session_id, database_session=database_session
        )
    except session_service.SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc



@stream_router.websocket("/sessions/{session_id}/stream")
async def stream_pose_frames(websocket: WebSocket, session_id: str):
    """Authenticated bidirectional stream of client-computed pose landmarks.

    The client sends an initial {"type": "authenticate", "access_token": "..."}
    message, then {"type": "frame", "frame": {...}} messages. Raw camera
    frames are intentionally not accepted by this endpoint.
    """
    await websocket.accept()
    factory = getattr(websocket.app.state, "database_session_factory", None)
    if factory is None:
        await websocket.close(code=1011, reason="Database is unavailable.")
        return

    try:
        auth_message = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        if not isinstance(auth_message, dict) or auth_message.get("type") != "authenticate":
            await websocket.close(code=4401, reason="Authentication required.")
            return

        session = factory()
        try:
            user = decode_token(session, str(auth_message.get("access_token", "")))
        except AuthenticationError:
            await websocket.close(code=4401, reason="Invalid or expired token.")
            return
        finally:
            session.close()

        if user.get("role") not in {"therapist", "admin"}:
            await websocket.close(code=4403, reason="Insufficient permissions.")
            return

        lifecycle_session = factory()
        try:
            from src.database.repositories import live_session_repository
            durable = live_session_repository.get_live_session(
                lifecycle_session, session_id
            )
            is_active = durable is not None and durable.status == "ACTIVE"
            if is_active and user.get("role") != "admin":
                from src.database.repositories import assignment_repository
                is_active = assignment_repository.therapist_has_patient(
                    lifecycle_session,
                    patient_id=durable.patient_id,
                    therapist_user_id=user["user_id"],
                )
        finally:
            lifecycle_session.close()

        if not is_active or not session_service.session_manager.has_session(session_id):
            await websocket.close(code=4404, reason="Active session not found.")
            return

        await websocket.send_json({"type": "authenticated", "user_id": user["user_id"]})

        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict) or message.get("type") != "frame":
                await websocket.send_json({
                    "type": "error",
                    "code": "invalid_message",
                    "detail": "Expected a frame message.",
                })
                continue
            try:
                frame = PoseFrameCreate.model_validate(message.get("frame", {}))
                analysis = await run_in_threadpool(
                    session_service.session_manager.add_frame, session_id, frame
                )
            except ValidationError as exc:
                await websocket.send_json({
                    "type": "error",
                    "code": "invalid_frame",
                    "detail": exc.errors(include_url=False),
                })
                continue
            except session_service.SessionValidationError as exc:
                await websocket.send_json({
                    "type": "error",
                    "code": "invalid_frame_sequence",
                    "detail": str(exc),
                })
                continue
            except session_service.SessionNotFoundError:
                await websocket.close(code=4404, reason="Active session not found.")
                return

            await websocket.send_json({"type": "analysis", "data": analysis})
    except asyncio.TimeoutError:
        await websocket.close(code=4401, reason="Authentication timed out.")
    except WebSocketDisconnect:
        return
