"""Application services wrapping the established NARS adaptive workflow."""

from src.adaptation import workflow
from src.services.patient_service import get_patient


def evaluate_session(patient_id, session_id, database_session=None):
    get_patient(patient_id, session=database_session)
    result = workflow.propose_after_session(
        patient_id, session_id, database_session=database_session
    )
    if result is None:
        raise workflow.AdaptationError(
            "The session does not match the patient's active supported prescription."
        )
    return result


def pending_proposals(patient_id, database_session=None):
    get_patient(patient_id)
    return workflow.pending_proposals(patient_id, database_session=database_session)


def decision_history(patient_id, database_session=None):
    get_patient(patient_id)
    return workflow.decision_history(patient_id, database_session=database_session)


def review_proposal(proposal_id, approve, reviewer, note="", database_session=None):
    status = workflow.review_proposal(
        proposal_id, approve=approve, reviewer=reviewer, note=note,
        database_session=database_session,
    )
    return {
        "proposal_id": proposal_id,
        "status": status,
        "reviewer": reviewer.strip(),
    }
