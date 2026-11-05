"""SQLAlchemy read repository for progress history and dashboard data."""

from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import ExerciseRecommendation, Patient, RehabSession


def list_patient_sessions(
    session: Session, patient_id: int, exercise_id: str | None = None,
    affected_side: str | None = None,
):
    statement = select(RehabSession).where(RehabSession.patient_id == patient_id)
    if exercise_id is not None:
        statement = statement.where(RehabSession.exercise_id == exercise_id)
    if affected_side is not None:
        statement = statement.where(RehabSession.affected_side == affected_side)
    rows = session.scalars(
        statement.order_by(RehabSession.session_date, RehabSession.session_id)
    )
    return [row.__dict__ | {} for row in rows]


def dashboard_counts(session: Session):
    now = datetime.now()
    return {
        "patients": session.scalar(select(func.count()).select_from(Patient)) or 0,
        "sessions": session.scalar(select(func.count()).select_from(RehabSession)) or 0,
        "sessions_last_7_days": session.scalar(
            select(func.count()).select_from(RehabSession).where(
                RehabSession.session_date >= now - timedelta(days=7)
            )
        ) or 0,
        "pending_adaptations": session.scalar(
            select(func.count()).select_from(ExerciseRecommendation).where(
                ExerciseRecommendation.source == "ADAPTATION",
                ExerciseRecommendation.status == "PENDING",
            )
        ) or 0,
        "last_session": session.scalar(select(func.max(RehabSession.session_date))),
    }


def recent_sessions(session: Session, limit: int = 10):
    statement = (
        select(RehabSession, Patient.name)
        .join(Patient, Patient.patient_id == RehabSession.patient_id)
        .order_by(RehabSession.session_date.desc(), RehabSession.session_id.desc())
        .limit(limit)
    )
    result = []
    for row, patient_name in session.execute(statement):
        result.append({
            "patient_id": row.patient_id,
            "patient_name": patient_name,
            **{key: value for key, value in row.__dict__.items() if not key.startswith("_")},
        })
    return result
