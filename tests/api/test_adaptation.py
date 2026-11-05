"""API contract tests for adaptive decision support and therapist review."""

from src.adaptation import workflow


def test_adaptation_routes_are_registered(api_client):
    response = api_client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/patients/{patient_id}/sessions/{session_id}/adaptation/evaluate" in paths
    assert "/api/v1/patients/{patient_id}/adaptation/pending" in paths
    assert "/api/v1/patients/{patient_id}/adaptation/history" in paths
    assert "/api/v1/adaptation/{proposal_id}/approve" in paths
    assert "/api/v1/adaptation/{proposal_id}/reject" in paths


def test_adaptation_evaluation_delegates_to_existing_workflow(
    api_client, monkeypatch
):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Adaptation Patient"}
    ).json()
    patient_id = patient["patient_id"]

    expected = {
        "action": "MAINTAIN",
        "feedback": "Continue at the same settings.",
        "decision": "MAINTAIN",
        "rationale": "Synthetic policy result",
        "decision_id": 12,
        "proposal_id": None,
        "previous_parameters": {"target_reps": 5},
        "proposed_parameters": {"target_reps": 5},
        "next_attempt_parameters": {"target_reps": 5},
    }
    monkeypatch.setattr(
        "src.services.adaptation_service.workflow.propose_after_session",
        lambda received_patient, received_session, **kwargs: expected
        if (received_patient, received_session) == (patient_id, 91)
        else None,
    )

    response = api_client.post(
        f"/api/v1/patients/{patient_id}/sessions/91/adaptation/evaluate"
    )
    assert response.status_code == 200
    assert response.json() == expected


def test_adaptation_evaluation_rejects_unmatched_session(api_client, monkeypatch):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Unmatched Patient"}
    ).json()
    monkeypatch.setattr(
        "src.services.adaptation_service.workflow.propose_after_session",
        lambda *_, **__: None,
    )

    response = api_client.post(
        f"/api/v1/patients/{patient['patient_id']}/sessions/999/adaptation/evaluate"
    )
    assert response.status_code == 409


def test_adaptation_review_requires_reviewer(api_client):
    response = api_client.post(
        "/api/v1/adaptation/5/approve",
        json={"reviewer": "   ", "note": "Review"},
    )
    assert response.status_code == 422


def test_adaptation_approval_and_rejection_delegate_to_workflow(
    api_client, monkeypatch
):
    calls = []

    def fake_review(proposal_id, approve, reviewer, note="", database_session=None):
        calls.append((proposal_id, approve, reviewer, note))
        return "APPROVED" if approve else "REJECTED"

    monkeypatch.setattr(workflow, "review_proposal", fake_review)

    approved = api_client.post(
        "/api/v1/adaptation/5/approve",
        json={"reviewer": "Synthetic Therapist", "note": "Approved in test"},
    )
    rejected = api_client.post(
        "/api/v1/adaptation/6/reject",
        json={"reviewer": "Synthetic Therapist", "note": "Rejected in test"},
    )

    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "REJECTED"
    assert calls == [
        (5, True, "Synthetic Therapist", "Approved in test"),
        (6, False, "Synthetic Therapist", "Rejected in test"),
    ]


def test_adaptation_pending_and_history_require_existing_patient(api_client):
    assert api_client.get("/api/v1/patients/999999/adaptation/pending").status_code == 404
    assert api_client.get("/api/v1/patients/999999/adaptation/history").status_code == 404
