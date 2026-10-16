"""
WS5: the full adaptive loop, offscreen with synthetic sessions.

session -> evaluation -> PENDING proposal -> therapist approval
        -> the NEXT session actually runs with the approved parameters.

Synthetic inputs only: this verifies the software loop, not clinical
appropriateness or webcam behaviour.
"""

import json

import pytest

import src.database.db as db
from src.adaptation.workflow import (
    AdaptationError,
    decision_history,
    pending_proposals,
    review_proposal,
)
from tests.helpers import perform, prescribe, seed_patient


@pytest.fixture
def abduction_patient(temp_db):
    patient_id, assessment_id = seed_patient()
    rec_id = prescribe(
        patient_id, assessment_id, "Shoulder Abduction",
        "shoulder_abduction", "LEFT",
        {"target_reps": 3, "target_range": 80},
    )
    return patient_id, rec_id


def test_full_loop_next_session_uses_approved_parameters(
    qapp, abduction_patient, scripted_camera
):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, original_rec = abduction_patient
    page = RehabPage()

    first = perform(page, scripted_camera, [84, 84, 84])
    assert first["decision"] == "INSUFFICIENT_DATA"
    assert first["proposal_id"] is None
    assert pending_proposals(patient_id) == []

    second = perform(page, scripted_camera, [84, 84, 84])
    assert second["decision"] == "INCREASE"
    assert second["proposed_parameters"] == {"target_reps": 5,
                                             "target_range": 80.0}
    assert "awaiting therapist approval" in page.session_status.text()

    # Not applied until approved.
    page.refresh()
    assert page.current_prescription_id == original_rec
    assert page.current_parameters["target_reps"] == 3

    [proposal] = pending_proposals(patient_id)
    assert review_proposal(proposal["recommendation_id"], True,
                           "Therapist A") == "APPROVED"

    page.refresh()
    assert page.current_prescription_id == proposal["recommendation_id"]
    assert page.current_parameters == {"target_reps": 5, "target_range": 80.0}

    perform(page, scripted_camera, [84] * 5)

    last = db.get_sessions(patient_id)[-1]
    assert last["prescription_id"] == proposal["recommendation_id"]
    assert json.loads(last["parameters_json"]) == {"target_range": 80.0,
                                                   "target_reps": 5}
    assert last["target_reps"] == 5
    assert last["reps"] == 5

    history = decision_history(patient_id)
    assert [h["decision"] for h in history] == [
        "INSUFFICIENT_DATA", "INCREASE", "INSUFFICIENT_DATA"
    ]
    increase = history[1]
    assert increase["proposal_status"] == "APPROVED"
    assert increase["reviewed_by"] == "Therapist A"
    assert json.loads(increase["previous_parameters_json"]) == {
        "target_range": 80.0, "target_reps": 3
    }
    assert json.loads(increase["inputs_json"])["sessions_used"]


def test_rejected_proposal_keeps_current_prescription(
    qapp, abduction_patient, scripted_camera
):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, original_rec = abduction_patient
    page = RehabPage()
    perform(page, scripted_camera, [84, 84, 84])
    perform(page, scripted_camera, [84, 84, 84])

    [proposal] = pending_proposals(patient_id)
    review_proposal(proposal["recommendation_id"], False, "Therapist A",
                    "Too soon")

    page.refresh()
    assert page.current_prescription_id == original_rec
    assert page.current_parameters["target_reps"] == 3

    with pytest.raises(AdaptationError):
        review_proposal(proposal["recommendation_id"], True, "Therapist A")


def test_stale_proposal_cannot_be_approved(
    qapp, abduction_patient, scripted_camera
):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, _ = abduction_patient
    page = RehabPage()
    perform(page, scripted_camera, [84, 84, 84])
    perform(page, scripted_camera, [84, 84, 84])
    [proposal] = pending_proposals(patient_id)

    # Therapist assigns a different exercise in the meantime.
    _, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Elbow Flexion", "elbow_flexion",
              "LEFT", {})

    with pytest.raises(AdaptationError, match="no longer active"):
        review_proposal(proposal["recommendation_id"], True, "Therapist A")


def test_reviewer_name_required(qapp, abduction_patient, scripted_camera):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, _ = abduction_patient
    page = RehabPage()
    perform(page, scripted_camera, [84, 84, 84])
    perform(page, scripted_camera, [84, 84, 84])
    [proposal] = pending_proposals(patient_id)

    with pytest.raises(AdaptationError):
        review_proposal(proposal["recommendation_id"], True, "  ")


def test_newer_proposal_supersedes_older_pending_one(
    qapp, abduction_patient, scripted_camera
):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, _ = abduction_patient
    page = RehabPage()
    for _ in range(3):
        perform(page, scripted_camera, [84, 84, 84])

    pending = pending_proposals(patient_id)
    assert len(pending) == 1
    statuses = [
        h["proposal_status"] for h in decision_history(patient_id)
        if h["proposal_status"]
    ]
    assert statuses.count("SUPERSEDED") == 1
