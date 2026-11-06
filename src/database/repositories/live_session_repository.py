"""Durable metadata for active and interrupted live rehabilitation sessions.

This repository does not persist the in-memory ExerciseAnalyzer. A process
restart marks ACTIVE rows INTERRUPTED; resuming a movement analyzer requires
an explicit client-side restart rather than pretending to restore lost state.
"""

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.database.models import LiveSession


def create_live_session(
    session: Session, *, session_id: str, patient_id: int,
    prescription_id: int, exercise_id: str, affected_side: str,
):
    row = LiveSession(
        session_id=session_id,
        patient_id=patient_id,
        prescription_id=prescription_id,
        exercise_id=exercise_id,
        affected_side=affected_side,
        status="ACTIVE",
        started_at=datetime.now(),
        frames_received=0,
    )
    session.add(row)
    session.commit()
    return row


def get_live_session(session: Session, session_id: str):
    return session.get(LiveSession, session_id)


def finish_live_session(
    session: Session, session_id: str, status: str, frames_received: int | None = None
):
    if status not in {"COMPLETED", "CANCELLED", "INTERRUPTED"}:
        raise ValueError("Invalid live-session terminal status.")
    values = {"status": status, "ended_at": datetime.now()}
    if frames_received is not None:
        values["frames_received"] = max(0, int(frames_received))
    result = session.execute(
        update(LiveSession)
        .where(LiveSession.session_id == session_id, LiveSession.status == "ACTIVE")
        .values(**values)
    )
    session.commit()
    return result.rowcount == 1


def increment_frame_count(session: Session, session_id: str):
    row = session.get(LiveSession, session_id)
    if row is None or row.status != "ACTIVE":
        return False
    row.frames_received += 1
    session.commit()
    return True


def mark_active_sessions_interrupted(session: Session):
    result = session.execute(
        update(LiveSession)
        .where(LiveSession.status == "ACTIVE")
        .values(status="INTERRUPTED", ended_at=datetime.now())
    )
    session.commit()
    return result.rowcount


def list_live_sessions(session: Session, patient_id: int):
    return list(session.scalars(
        select(LiveSession)
        .where(LiveSession.patient_id == patient_id)
        .order_by(LiveSession.started_at.desc())
    ))
