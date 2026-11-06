"""WebSocket authentication and lifecycle tests for pose streaming."""

import pytest
from fastapi import WebSocketDisconnect

from src.services import auth_service


def _token(api_client, username="stream-therapist", patient_id=None):
    from src.database.repositories import assignment_repository

    session = api_client.app.state.database_session_factory()
    try:
        user = auth_service.create_user(
            session, username, "long-test-password-123", "therapist"
        )
        if patient_id is not None:
            assignment_repository.assign_therapist(
                session, patient_id=patient_id,
                therapist_user_id=user["user_id"], assigned_by_user_id=2,
            )
        return auth_service.issue_token(user)["access_token"]
    finally:
        session.close()


def test_stream_rejects_invalid_token(api_client):
    with pytest.raises(WebSocketDisconnect) as closed:
        with api_client.websocket_connect("/api/v1/sessions/not-a-session/stream") as ws:
            ws.send_json({"type": "authenticate", "access_token": "invalid"})
            ws.receive_json()
    assert closed.value.code == 4401


def test_stream_requires_authentication_message(api_client):
    with pytest.raises(WebSocketDisconnect) as closed:
        with api_client.websocket_connect("/api/v1/sessions/not-a-session/stream") as ws:
            ws.send_json({"type": "unexpected"})
            ws.receive_json()
    assert closed.value.code == 4401


def test_stream_rejects_unknown_session_after_authentication(api_client):
    token = _token(api_client)
    with pytest.raises(WebSocketDisconnect) as closed:
        with api_client.websocket_connect("/api/v1/sessions/not-a-session/stream") as ws:
            ws.send_json({"type": "authenticate", "access_token": token})
            ws.receive_json()
    assert closed.value.code == 4404


def test_stream_returns_analysis_for_active_session(api_client):
    # Create a patient and approved exercise prescription through the public API.
    patient = api_client.post(
        "/api/v1/patients", json={"name": "WebSocket Stream Patient"}
    ).json()
    patient_id = patient["patient_id"]
    api_client.post(
        f"/api/v1/patients/{patient_id}/affected-areas",
        json={"side": "LEFT", "body_region": "Upper Limb", "joint": "Shoulder"},
    )
    assessment = api_client.post(
        f"/api/v1/patients/{patient_id}/assessments",
        json={
            "diagnosis": "Stroke",
            "impairment": "Reduced ROM",
            "baseline_rom": 80,
        },
    ).json()
    recommendation = api_client.post(
        f"/api/v1/patients/{patient_id}/recommendations/generate",
        json={"assessment_id": assessment["assessment_id"]},
    ).json()
    from src.exercises.catalogue import get_exercise_config
    config = get_exercise_config(recommendation["exercise_id"])
    approved = api_client.post(
        f"/api/v1/recommendations/{recommendation['recommendation_id']}/approve",
        json={
            "reviewer": "Stream Test Therapist",
            "exercise_id": config.exercise_id,
            "parameters": config.default_parameters(),
        },
    )
    assert approved.status_code == 200

    started = api_client.post(
        "/api/v1/sessions", json={"patient_id": patient_id}
    )
    assert started.status_code == 201
    session_id = started.json()["session_id"]
    token = _token(api_client, "streaming-therapist", patient_id=patient_id)

    with api_client.websocket_connect(
        f"/api/v1/sessions/{session_id}/stream"
    ) as ws:
        ws.send_json({"type": "authenticate", "access_token": token})
        assert ws.receive_json()["type"] == "authenticated"
        ws.send_json({
            "type": "frame",
            "frame": {"timestamp_s": 0.0, "pose_landmarks": None},
        })
        result = ws.receive_json()
        assert result["type"] == "analysis"
        assert result["data"]["status"] == "no_pose"
        assert result["data"]["reps"] == 0

    cancelled = api_client.delete(f"/api/v1/sessions/{session_id}")
    assert cancelled.status_code == 200
