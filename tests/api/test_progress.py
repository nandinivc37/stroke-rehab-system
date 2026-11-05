"""API tests for read-only progress and dashboard endpoints."""


def test_dashboard_summary_returns_database_counts(api_client):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Dashboard Patient"}
    )
    assert patient.status_code == 201

    response = api_client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["counts"]["patients"] == 1
    assert payload["counts"]["sessions"] == 0
    assert payload["recent_sessions"] == []


def test_patient_progress_returns_empty_history_for_new_patient(api_client):
    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Progress Patient"}
    ).json()
    patient_id = patient["patient_id"]

    response = api_client.get(f"/api/v1/patients/{patient_id}/progress")
    assert response.status_code == 200
    assert response.json() == {
        "patient_id": patient_id,
        "session_count": 0,
        "groups": [],
        "sessions": [],
    }


def test_progress_endpoints_return_404_for_unknown_patient(api_client):
    assert api_client.get(
        "/api/v1/patients/999999/progress"
    ).status_code == 404
    assert api_client.get(
        "/api/v1/patients/999999/progress/trends"
    ).status_code == 404


def test_progress_trends_preserve_legacy_group_separation(api_client, monkeypatch):
    from src.services import progress_service

    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Trend Patient"}
    ).json()
    patient_id = patient["patient_id"]

    legacy = {
        "session_id": 3,
        "session_date": "2026-09-01",
        "exercise": "Shoulder Flexion",
        "exercise_id": None,
        "affected_side": None,
        "metrics_version": None,
        "reps": 4,
        "target_reps": None,
        "max_angle": 65.0,
        "peak_rom": None,
        "trajectory_similarity": None,
        "dtw_similarity": None,
        "valid_segments": None,
        "frames_total": None,
        "frames_measured": None,
        "mean_fps": None,
        "session_status": "COMPLETE",
    }
    current = {
        **legacy,
        "session_id": 4,
        "session_date": "2026-10-01",
        "exercise_id": "shoulder_flexion",
        "affected_side": "LEFT",
        "metrics_version": 2,
        "peak_rom": 70.0,
        "trajectory_similarity": 82.0,
        "dtw_similarity": 90.0,
        "frames_total": 100,
        "frames_measured": 90,
    }
    monkeypatch.setattr(
        progress_service.db,
        "get_sessions",
        lambda requested_patient, **kwargs: [legacy, current]
        if requested_patient == patient_id else [],
    )

    response = api_client.get(
        f"/api/v1/patients/{patient_id}/progress/trends"
    )
    assert response.status_code == 200
    groups = response.json()["groups"]
    assert len(groups) == 2
    assert groups[0]["legacy_metrics"] is False
    assert groups[1]["legacy_metrics"] is True
    assert groups[0]["points"][0]["peak_value"] == 70.0
    assert groups[1]["points"][0]["peak_value"] == 65.0
