"""Tests for landmark-based rehabilitation session API."""

from src.database import db
from src.exercises.catalogue import get_exercise_config


def create_patient_with_approved_prescription(api_client):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Session Patient"}
    )
    assert patient.status_code == 201
    patient_id = patient.json()["patient_id"]

    area = api_client.post(
        f"/api/v1/patients/{patient_id}/affected-areas",
        json={
            "side": "LEFT",
            "body_region": "Upper Limb",
            "joint": "Shoulder",
        },
    )
    assert area.status_code == 201

    assessment = api_client.post(
        f"/api/v1/patients/{patient_id}/assessments",
        json={
            "diagnosis": "Stroke",
            "impairment": "Reduced ROM",
            "baseline_rom": 80,
        },
    )
    assert assessment.status_code == 201

    recommendation = api_client.post(
        f"/api/v1/patients/{patient_id}/recommendations/generate",
        json={"assessment_id": assessment.json()["assessment_id"]},
    )
    assert recommendation.status_code == 200
    proposed = recommendation.json()
    config = get_exercise_config(proposed["exercise_id"])
    assert config is not None

    approved = api_client.post(
        f"/api/v1/recommendations/{proposed['recommendation_id']}/approve",
        json={
            "reviewer": "Synthetic Therapist",
            "exercise_id": config.exercise_id,
            "parameters": config.default_parameters(),
        },
    )
    assert approved.status_code == 200
    return patient_id


def test_session_requires_approved_prescription(api_client):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "No Prescription Patient"}
    ).json()

    response = api_client.post(
        "/api/v1/sessions", json={"patient_id": patient["patient_id"]}
    )

    assert response.status_code == 409
    assert "therapist-approved" in response.json()["detail"]


def test_session_accepts_no_pose_frame_and_persists_summary(api_client):
    patient_id = create_patient_with_approved_prescription(api_client)

    started = api_client.post(
        "/api/v1/sessions", json={"patient_id": patient_id}
    )
    assert started.status_code == 201
    session_id = started.json()["session_id"]
    assert started.json()["status"] == "ACTIVE"

    frame = api_client.post(
        f"/api/v1/sessions/{session_id}/frames",
        json={"timestamp_s": 0.0, "pose_landmarks": None},
    )
    assert frame.status_code == 200
    assert frame.json()["status"] == "no_pose"
    assert frame.json()["reps"] == 0

    completed = api_client.post(
        f"/api/v1/sessions/{session_id}/complete"
    )
    assert completed.status_code == 200
    result = completed.json()
    assert result["status"] == db.SESSION_STATUS_COMPLETE
    assert result["patient_id"] == patient_id
    assert result["summary"]["frames_total"] == 1
    assert result["summary"]["frames_with_pose"] == 0

    stored = db.get_sessions(patient_id)
    assert len(stored) == 1
    assert stored[0]["session_status"] == db.SESSION_STATUS_COMPLETE
    assert stored[0]["metrics_version"] is not None

    stored_frames = db.get_session_frames(result["session_id"])
    assert len(stored_frames) == 1
    assert stored_frames[0]["frame_index"] == 0
    assert stored_frames[0]["status"] == "no_pose"


def test_session_rejects_non_increasing_timestamps(api_client):
    patient_id = create_patient_with_approved_prescription(api_client)
    started = api_client.post(
        "/api/v1/sessions", json={"patient_id": patient_id}
    ).json()
    session_id = started["session_id"]

    first = api_client.post(
        f"/api/v1/sessions/{session_id}/frames",
        json={"timestamp_s": 1.0},
    )
    second = api_client.post(
        f"/api/v1/sessions/{session_id}/frames",
        json={"timestamp_s": 1.0},
    )

    assert first.status_code == 200
    assert second.status_code == 422

    cancelled = api_client.delete(f"/api/v1/sessions/{session_id}")
    assert cancelled.status_code == 200


def test_session_rejects_wrong_landmark_count(api_client):
    patient_id = create_patient_with_approved_prescription(api_client)
    started = api_client.post(
        "/api/v1/sessions", json={"patient_id": patient_id}
    ).json()
    session_id = started["session_id"]

    response = api_client.post(
        f"/api/v1/sessions/{session_id}/frames",
        json={
            "timestamp_s": 0.0,
            "pose_landmarks": [{"x": 0.1, "y": 0.2}] * 5,
        },
    )

    assert response.status_code == 422
    api_client.delete(f"/api/v1/sessions/{session_id}")



def test_live_session_lifecycle_is_persisted(api_client):
    from src.database.repositories import live_session_repository

    patient_id = create_patient_with_approved_prescription(api_client)
    started = api_client.post(
        "/api/v1/sessions", json={"patient_id": patient_id}
    )
    assert started.status_code == 201
    session_id = started.json()["session_id"]

    factory = api_client.app.state.database_session_factory
    session = factory()
    try:
        row = live_session_repository.get_live_session(session, session_id)
        assert row is not None
        assert row.patient_id == patient_id
        assert row.status == "ACTIVE"
        assert row.frames_received == 0
    finally:
        session.close()

    cancelled = api_client.delete(f"/api/v1/sessions/{session_id}")
    assert cancelled.status_code == 200

    session = factory()
    try:
        row = live_session_repository.get_live_session(session, session_id)
        assert row.status == "CANCELLED"
        assert row.ended_at is not None
    finally:
        session.close()


def test_completed_live_session_has_durable_terminal_status(api_client):
    from src.database.repositories import live_session_repository

    patient_id = create_patient_with_approved_prescription(api_client)
    started = api_client.post(
        "/api/v1/sessions", json={"patient_id": patient_id}
    ).json()
    session_id = started["session_id"]

    frame = api_client.post(
        f"/api/v1/sessions/{session_id}/frames",
        json={"timestamp_s": 0.0, "pose_landmarks": None},
    )
    assert frame.status_code == 200
    completed = api_client.post(f"/api/v1/sessions/{session_id}/complete")
    assert completed.status_code == 200

    session = api_client.app.state.database_session_factory()
    try:
        row = live_session_repository.get_live_session(session, session_id)
        assert row.status == "COMPLETED"
        assert row.frames_received == 1
        assert row.ended_at is not None
    finally:
        session.close()



def test_api_session_start_uses_configured_orm_database(api_client, monkeypatch):
    patient_id = create_patient_with_approved_prescription(api_client)

    def legacy_access_forbidden(*args, **kwargs):
        raise AssertionError("API session start must not read the legacy SQLite layer.")

    monkeypatch.setattr(db, "get_patient", legacy_access_forbidden)
    monkeypatch.setattr(db, "get_active_prescription", legacy_access_forbidden)
    monkeypatch.setattr(db, "get_affected_areas", legacy_access_forbidden)

    response = api_client.post(
        "/api/v1/sessions", json={"patient_id": patient_id}
    )
    assert response.status_code == 201
    assert response.json()["status"] == "ACTIVE"
    api_client.delete(f"/api/v1/sessions/{response.json()['session_id']}")



def test_active_session_capacity_is_enforced(api_client, monkeypatch):
    from src.services import session_service

    patient_id = create_patient_with_approved_prescription(api_client)
    manager = session_service.session_manager
    original_limit = manager.max_active_sessions
    manager.max_active_sessions = 1
    try:
        first = api_client.post("/api/v1/sessions", json={"patient_id": patient_id})
        assert first.status_code == 201

        second = api_client.post("/api/v1/sessions", json={"patient_id": patient_id})
        assert second.status_code == 409
        assert "active rehabilitation session limit" in second.json()["detail"]

        cancelled = api_client.delete(
            f"/api/v1/sessions/{first.json()['session_id']}"
        )
        assert cancelled.status_code == 200
    finally:
        manager.max_active_sessions = original_limit


def test_idle_session_is_interrupted_during_next_start(api_client):
    from time import monotonic
    from src.services import session_service
    from src.database.repositories import live_session_repository

    patient_id = create_patient_with_approved_prescription(api_client)
    first = api_client.post("/api/v1/sessions", json={"patient_id": patient_id})
    assert first.status_code == 201
    first_id = first.json()["session_id"]

    manager = session_service.session_manager
    manager._sessions[first_id].last_activity_monotonic = (
        monotonic() - manager.idle_timeout_seconds - 1
    )

    second = api_client.post("/api/v1/sessions", json={"patient_id": patient_id})
    assert second.status_code == 201

    session = api_client.app.state.database_session_factory()
    try:
        row = live_session_repository.get_live_session(session, first_id)
        assert row.status == "INTERRUPTED"
        assert row.ended_at is not None
    finally:
        session.close()
        api_client.delete(f"/api/v1/sessions/{second.json()['session_id']}")
