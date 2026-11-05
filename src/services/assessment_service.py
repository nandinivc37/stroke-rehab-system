"""Clinical assessment and recommendation services.

Clinical decision rules remain in src.assessment.recommendation. This layer
orchestrates persistence while preserving mandatory therapist review.
"""

from src.assessment.recommendation import recommend_exercise, override_note
from src.database.repositories import assessment_repository, patient_repository
from src.exercises.catalogue import get_exercise_config
from src.services.patient_service import get_patient


class AssessmentNotFoundError(LookupError):
    """Requested assessment does not exist for the specified patient."""


class RecommendationNotFoundError(LookupError):
    """Requested recommendation does not exist."""


class RecommendationConflictError(RuntimeError):
    """Recommendation is no longer awaiting therapist review."""


def _assessment_dict(row, patient_id):
    if row is None:
        return None
    return {
        "assessment_id": row.assessment_id,
        "patient_id": patient_id,
        "diagnosis": row.diagnosis,
        "impairment": row.impairment,
        "baseline_rom": row.baseline_rom,
        "restrictions": row.restrictions,
        "clinical_notes": row.clinical_notes,
        "assessment_date": row.assessment_date.isoformat() if row.assessment_date else None,
    }


def create_assessment(session, patient_id, payload):
    get_patient(patient_id, session=session)
    if not patient_repository.list_affected_areas(session, patient_id):
        raise ValueError("Record an affected area before creating an assessment.")
    row = assessment_repository.create_assessment(
        session, patient_id=patient_id, payload=payload
    )
    return _assessment_dict(row, patient_id)


def get_latest_assessment(session, patient_id):
    get_patient(patient_id, session=session)
    row = assessment_repository.get_latest_assessment(session, patient_id)
    if row is None:
        raise AssessmentNotFoundError("No clinical assessment exists for this patient.")
    return _assessment_dict(row, patient_id)


def generate_recommendation(session, patient_id, assessment_id):
    get_patient(patient_id, session=session)
    row = assessment_repository.get_latest_assessment(session, patient_id)
    if row is None or row.assessment_id != assessment_id:
        raise AssessmentNotFoundError("Assessment not found for this patient.")
    areas = patient_repository.list_affected_areas(session, patient_id)
    if not areas:
        raise ValueError("Patient has no affected area recorded.")
    area = areas[0]
    result = recommend_exercise(
        diagnosis=row.diagnosis, affected_side=area.side,
        body_region=area.body_region, joint=area.joint,
        impairment=row.impairment, baseline_rom=row.baseline_rom,
        restrictions=row.restrictions,
    )
    if not result["success"]:
        return result
    recommendation = assessment_repository.create_recommendation(
        session, patient_id=patient_id, assessment_id=assessment_id,
        exercise=result["exercise"], difficulty=result["difficulty"],
        reason=result["reason"], exercise_id=result["exercise_id"],
        affected_side=area.side, parameters=result["parameters"],
    )
    result.update({
        "recommendation_id": recommendation["recommendation_id"],
        "patient_id": patient_id,
        "assessment_id": assessment_id,
        "status": "PENDING",
    })
    return result


def list_recommendations(session, patient_id, status=None):
    get_patient(patient_id, session=session)
    return assessment_repository.list_recommendations(session, patient_id, status=status)


def review_recommendation(session, recommendation_id, payload, approve=True):
    recommendation = assessment_repository.get_recommendation(session, recommendation_id)
    if recommendation is None:
        raise RecommendationNotFoundError("Recommendation not found.")
    if recommendation["status"] != "PENDING":
        raise RecommendationConflictError("Only pending recommendations can be reviewed.")
    reviewer = payload.reviewer.strip()
    if not approve:
        try:
            return assessment_repository.review_recommendation(
                session, recommendation_id, status="REJECTED",
                reviewer=reviewer, note=payload.note,
            )
        except RuntimeError as exc:
            raise RecommendationConflictError(str(exc)) from exc

    if not payload.exercise_id or payload.parameters is None:
        raise ValueError("Exercise and starting parameters are required for approval.")
    config = get_exercise_config(payload.exercise_id)
    if config is None:
        raise ValueError("Selected exercise is not implemented.")
    approved_parameters = config.resolve_parameters(payload.parameters)
    note = override_note(
        recommendation["exercise"], recommendation.get("parameters") or {},
        config, approved_parameters,
    )
    reason = recommendation.get("reason") or ""
    if note:
        reason = f"{reason} {note}".strip()
    try:
        return assessment_repository.review_recommendation(
            session, recommendation_id, status="APPROVED", reviewer=reviewer,
            note=payload.note, exercise=config.display_name,
            exercise_id=config.exercise_id,
            affected_side=recommendation.get("affected_side"),
            parameters=approved_parameters, reason=reason or None,
            difficulty="Therapist-set start" if note else None,
        )
    except RuntimeError as exc:
        raise RecommendationConflictError(str(exc)) from exc
