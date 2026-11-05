"""Clinical assessment and recommendation application services."""

from src.assessment.recommendation import recommend_exercise, override_note
from src.database import db
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
        "assessment_id": row["assessment_id"],
        "patient_id": patient_id,
        "diagnosis": row["diagnosis"],
        "impairment": row["impairment"],
        "baseline_rom": row["baseline_rom"],
        "restrictions": row["restrictions"],
        "clinical_notes": row["clinical_notes"],
        "assessment_date": str(row["assessment_date"]),
    }


def create_assessment(patient_id, payload):
    get_patient(patient_id)
    if not db.get_affected_areas(patient_id):
        raise ValueError("Record an affected area before creating an assessment.")
    assessment_id = db.save_clinical_assessment(
        patient_id, payload.diagnosis.strip(), payload.impairment.strip(),
        payload.baseline_rom, payload.restrictions, payload.clinical_notes,
    )
    row = db.get_current_assessment(patient_id)
    return _assessment_dict(row, patient_id)


def get_latest_assessment(patient_id):
    get_patient(patient_id)
    row = db.get_current_assessment(patient_id)
    if row is None:
        raise AssessmentNotFoundError("No clinical assessment exists for this patient.")
    return _assessment_dict(row, patient_id)


def generate_recommendation(patient_id, assessment_id):
    get_patient(patient_id)
    row = db.get_current_assessment(patient_id)
    if row is None or row["assessment_id"] != assessment_id:
        raise AssessmentNotFoundError("Assessment not found for this patient.")
    areas = db.get_affected_areas(patient_id)
    if not areas:
        raise ValueError("Patient has no affected area recorded.")
    area = areas[0]
    result = recommend_exercise(
        diagnosis=row["diagnosis"], affected_side=area[1],
        body_region=area[2], joint=area[3], impairment=row["impairment"],
        baseline_rom=row["baseline_rom"], restrictions=row["restrictions"],
    )
    if not result["success"]:
        return result
    recommendation_id = db.save_recommendation(
        patient_id, assessment_id, result["exercise"],
        result["difficulty"], result["reason"],
    )
    db.set_recommendation_prescription(
        recommendation_id, result["exercise"], result["exercise_id"],
        area[1], result["parameters"], "ASSESSMENT",
    )
    result.update({
        "recommendation_id": recommendation_id,
        "patient_id": patient_id,
        "assessment_id": assessment_id,
        "status": "PENDING",
    })
    return result


def review_recommendation(recommendation_id, payload, approve=True):
    recommendation = db.get_recommendation(recommendation_id)
    if recommendation is None:
        raise RecommendationNotFoundError("Recommendation not found.")
    if recommendation["status"] != "PENDING":
        raise RecommendationConflictError("Only pending recommendations can be reviewed.")

    if not approve:
        db.record_recommendation_review(
            recommendation_id, "REJECTED", payload.reviewer.strip(),
            payload.note,
        )
        return db.get_recommendation(recommendation_id)

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
    reason = (recommendation.get("reason") or "")
    if note:
        reason = f"{reason} {note}".strip()
    db.set_recommendation_prescription(
        recommendation_id, config.display_name, config.exercise_id,
        recommendation.get("affected_side"), approved_parameters,
        "ASSESSMENT", reason=reason or None,
        difficulty="Therapist-set start" if note else None,
    )
    db.record_recommendation_review(
        recommendation_id, "APPROVED", payload.reviewer.strip(),
        payload.note,
    )
    return db.get_recommendation(recommendation_id)
