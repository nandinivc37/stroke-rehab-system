"""SQLAlchemy repository for adaptive decisions and therapist proposals."""

import json
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.database.models import AdaptationDecision, ExerciseRecommendation


def record_decision(
    session: Session, *, patient_id: int, exercise_id: str, affected_side: str,
    session_id: int, policy_version: str, decision: str, rationale: str,
    inputs: dict, previous_parameters: dict, proposed_parameters: dict,
    previous_recommendation_id: int, proposal: dict | None = None,
):
    """Atomically record a decision and, when needed, its pending proposal."""
    row = AdaptationDecision(
        patient_id=patient_id, exercise_id=exercise_id,
        affected_side=affected_side, session_id=session_id,
        policy_version=policy_version, decision=decision, rationale=rationale,
        inputs_json=json.dumps(inputs, sort_keys=True, default=str),
        previous_parameters_json=json.dumps(previous_parameters, sort_keys=True),
        proposed_parameters_json=json.dumps(proposed_parameters, sort_keys=True),
        previous_recommendation_id=previous_recommendation_id,
        created_at=datetime.now(),
    )
    session.add(row)
    session.flush()
    proposal_id = None
    if proposal is not None:
        session.execute(
            update(ExerciseRecommendation)
            .where(
                ExerciseRecommendation.patient_id == patient_id,
                ExerciseRecommendation.source == "ADAPTATION",
                ExerciseRecommendation.status == "PENDING",
            )
            .values(status="SUPERSEDED")
        )
        recommendation = ExerciseRecommendation(
            patient_id=patient_id, assessment_id=proposal["assessment_id"],
            exercise=proposal["exercise"], difficulty=proposal["difficulty"],
            reason=proposal["reason"], status="PENDING",
            created_at=datetime.now(), exercise_id=exercise_id,
            affected_side=affected_side,
            parameters_json=json.dumps(proposed_parameters, sort_keys=True),
            source="ADAPTATION", adaptation_decision_id=row.decision_id,
        )
        session.add(recommendation)
        session.flush()
        proposal_id = recommendation.recommendation_id
        row.proposed_recommendation_id = proposal_id
    session.commit()
    return row.decision_id, proposal_id


def list_pending_proposals(session: Session, patient_id: int):
    statement = (
        select(ExerciseRecommendation, AdaptationDecision)
        .join(
            AdaptationDecision,
            AdaptationDecision.decision_id == ExerciseRecommendation.adaptation_decision_id,
        )
        .where(
            ExerciseRecommendation.patient_id == patient_id,
            ExerciseRecommendation.source == "ADAPTATION",
            ExerciseRecommendation.status == "PENDING",
        )
        .order_by(ExerciseRecommendation.recommendation_id.desc())
    )
    result = []
    for recommendation, decision in session.execute(statement):
        result.append({
            "recommendation_id": recommendation.recommendation_id,
            "exercise": recommendation.exercise,
            "exercise_id": recommendation.exercise_id,
            "affected_side": recommendation.affected_side,
            "parameters_json": recommendation.parameters_json,
            "reason": recommendation.reason,
            "created_at": recommendation.created_at,
            "decision": decision.decision,
            "previous_parameters_json": decision.previous_parameters_json,
            "previous_recommendation_id": decision.previous_recommendation_id,
        })
    return result


def list_decision_history(session: Session, patient_id: int):
    statement = (
        select(AdaptationDecision, ExerciseRecommendation.status, ExerciseRecommendation.reviewed_by)
        .outerjoin(
            ExerciseRecommendation,
            ExerciseRecommendation.recommendation_id == AdaptationDecision.proposed_recommendation_id,
        )
        .where(AdaptationDecision.patient_id == patient_id)
        .order_by(AdaptationDecision.decision_id.desc())
    )
    rows = []
    for decision, proposal_status, reviewed_by in session.execute(statement):
        row = {
            "decision_id": decision.decision_id,
            "patient_id": decision.patient_id,
            "exercise_id": decision.exercise_id,
            "affected_side": decision.affected_side,
            "session_id": decision.session_id,
            "policy_version": decision.policy_version,
            "decision": decision.decision,
            "rationale": decision.rationale,
            "inputs_json": decision.inputs_json,
            "previous_parameters_json": decision.previous_parameters_json,
            "proposed_parameters_json": decision.proposed_parameters_json,
            "previous_recommendation_id": decision.previous_recommendation_id,
            "proposed_recommendation_id": decision.proposed_recommendation_id,
            "created_at": decision.created_at,
            "proposal_status": proposal_status,
            "reviewed_by": reviewed_by,
        }
        try:
            controller = json.loads(row["inputs_json"] or "{}").get("controller") or {}
        except (TypeError, ValueError, AttributeError):
            controller = {}
        row["action"] = controller.get("action")
        row["feedback"] = controller.get("feedback")
        rows.append(row)
    return rows


def get_adaptation_proposal(session: Session, proposal_id: int):
    statement = (
        select(ExerciseRecommendation, AdaptationDecision.previous_recommendation_id)
        .join(
            AdaptationDecision,
            AdaptationDecision.decision_id == ExerciseRecommendation.adaptation_decision_id,
        )
        .where(
            ExerciseRecommendation.recommendation_id == proposal_id,
            ExerciseRecommendation.source == "ADAPTATION",
        )
    )
    row = session.execute(statement).first()
    if row is None:
        return None
    recommendation, based_on = row
    return {
        "patient_id": recommendation.patient_id,
        "status": recommendation.status,
        "previous_recommendation_id": based_on,
    }


def review_proposal(session: Session, proposal_id: int, *, status: str, reviewer: str, note: str = ""):
    if status not in {"APPROVED", "REJECTED"}:
        raise ValueError("Review status must be APPROVED or REJECTED.")
    values = {"status": status, "reviewed_by": reviewer.strip(), "review_note": note}
    if status == "APPROVED":
        values["approved_at"] = datetime.now()
    result = session.execute(
        update(ExerciseRecommendation)
        .where(
            ExerciseRecommendation.recommendation_id == proposal_id,
            ExerciseRecommendation.source == "ADAPTATION",
            ExerciseRecommendation.status == "PENDING",
        )
        .values(**values)
    )
    if result.rowcount != 1:
        session.rollback()
        raise RuntimeError("Adaptation proposal is no longer pending.")
    session.commit()
    return status
