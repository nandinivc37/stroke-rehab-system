"""Application services wrapping the established NARS adaptive workflow."""

from src.adaptation import workflow
from src.services.patient_service import get_patient, PatientNotFoundError


def evaluate_session(patient_id, session_id):
    get_patient(patient_id)
    result = workflow.propose_after_session(patient_id, session_id)
    if result is None:
        raise workflow.AdaptationError(
            "The session does not match the patient's active supported prescription."
        )
    return result


def pending_proposals(patient_id):
    get_patient(patient_id)
    return workflow.pending_proposals(patient_id)


def decision_history(patient_id):
    get_patient(patient_id)
    return workflow.decision_history(patient_id)


def review_proposal(proposal_id, approve, reviewer, note=""):
    status = workflow.review_proposal(
        proposal_id, approve=approve, reviewer=reviewer, note=note
    )
    return {
        "proposal_id": proposal_id,
        "status": status,
        "reviewer": reviewer.strip(),
    }
