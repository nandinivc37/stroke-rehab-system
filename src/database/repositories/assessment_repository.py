"""SQLAlchemy repository for clinical assessments and exercise recommendations."""

import json
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.database.models import ClinicalAssessment, ExerciseRecommendation


def create_assessment(session: Session, *, patient_id: int, payload):
    row = ClinicalAssessment(
        patient_id=patient_id,
        diagnosis=payload.diagnosis.strip(),
        impairment=payload.impairment.strip(),
        baseline_rom=payload.baseline_rom,
        restrictions=payload.restrictions,
        clinical_notes=payload.clinical_notes,
        assessment_date=datetime.now(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_latest_assessment(session: Session, patient_id: int):
    return session.scalars(
        select(ClinicalAssessment)
        .where(ClinicalAssessment.patient_id == patient_id)
        .order_by(ClinicalAssessment.assessment_date.desc(), ClinicalAssessment.assessment_id.desc())
        .limit(1)
    ).first()


def create_recommendation(
    session: Session, *, patient_id: int, assessment_id: int,
    exercise: str, difficulty: str, reason: str, exercise_id: str,
    affected_side: str, parameters: dict, source: str = "ASSESSMENT",
):
    row = ExerciseRecommendation(
        patient_id=patient_id,
        assessment_id=assessment_id,
        exercise=exercise,
        difficulty=difficulty,
        reason=reason,
        status="PENDING",
        created_at=datetime.now(),
        exercise_id=exercise_id,
        affected_side=affected_side,
        parameters_json=json.dumps(parameters, sort_keys=True),
        source=source,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return recommendation_dict(row)


def _recommendation_dict(row):
    return {
        "recommendation_id": row.recommendation_id,
        "patient_id": row.patient_id,
        "assessment_id": row.assessment_id,
        "exercise": row.exercise,
        "exercise_id": row.exercise_id,
        "difficulty": row.difficulty,
        "reason": row.reason,
        "status": row.status,
        "created_at": row.created_at,
        "approved_at": row.approved_at,
        "affected_side": row.affected_side,
        "parameters": json.loads(row.parameters_json) if row.parameters_json else {},
        "source": row.source,
        "reviewed_by": row.reviewed_by,
        "review_note": row.review_note,
    }


def recommendation_dict(row):
    return _recommendation_dict(row)


def get_recommendation(session: Session, recommendation_id: int):
    row = session.get(ExerciseRecommendation, recommendation_id)
    return _recommendation_dict(row) if row is not None else None


def list_recommendations(session: Session, patient_id: int, status: str | None = None):
    statement = select(ExerciseRecommendation).where(
        ExerciseRecommendation.patient_id == patient_id
    )
    if status is not None:
        statement = statement.where(ExerciseRecommendation.status == status)
    rows = session.scalars(
        statement.order_by(ExerciseRecommendation.recommendation_id.desc())
    )
    return [_recommendation_dict(row) for row in rows]


def review_recommendation(
    session: Session, recommendation_id: int, *, status: str, reviewer: str,
    note: str | None = None, exercise: str | None = None,
    exercise_id: str | None = None, affected_side: str | None = None,
    parameters: dict | None = None, reason: str | None = None,
    difficulty: str | None = None,
):
    if status not in {"APPROVED", "REJECTED"}:
        raise ValueError("Review status must be APPROVED or REJECTED.")
    values = {
        "status": status,
        "reviewed_by": reviewer.strip(),
        "review_note": note,
    }
    if status == "APPROVED":
        if not exercise or not exercise_id or parameters is None:
            raise ValueError("Approved prescriptions require an exercise and parameters.")
        values.update({
            "exercise": exercise,
            "exercise_id": exercise_id,
            "affected_side": affected_side,
            "parameters_json": json.dumps(parameters, sort_keys=True),
            "approved_at": datetime.now(),
        })
        if reason is not None:
            values["reason"] = reason
        if difficulty is not None:
            values["difficulty"] = difficulty
    result = session.execute(
        update(ExerciseRecommendation)
        .where(
            ExerciseRecommendation.recommendation_id == recommendation_id,
            ExerciseRecommendation.status == "PENDING",
        )
        .values(**values)
    )
    if result.rowcount != 1:
        session.rollback()
        raise RuntimeError("Only pending recommendations can be reviewed.")
    session.commit()
    row = session.get(ExerciseRecommendation, recommendation_id)
    return _recommendation_dict(row)
