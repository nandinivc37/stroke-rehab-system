"""API tests for read-only progress and dashboard endpoints."""

from src.api.dependencies import get_current_user


def test_dashboard_summary_returns_database_counts(api_client):
    api_client.app.dependency_overrides[get_current_user] = lambda: {
        "user_id": 2, "username": "test-admin", "role": "admin", "is_active": True,
    }
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


def test_progress_trends_preserve_legacy_group_separation(api_client):
    from src.database import db

    patient = api_client.post(
        "/api/v1/patients", json={"name": "Synthetic Trend Patient"}
    ).json()
    patient_id = patient["patient_id"]

    conn = db.get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO sessions(
                patient_id, exercise, reps, max_angle, session_date,
                exercise_id, affected_side, metrics_version, peak_rom,
                trajectory_similarity, dtw_similarity, frames_total,
                frames_measured, session_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    patient_id, "Shoulder Flexion", 4, 65.0, "2026-09-01",
                    None, None, None, None, None, None, None, None, "COMPLETE",
                ),
                (
                    patient_id, "Shoulder Flexion", 4, 70.0, "2026-10-01",
                    "shoulder_flexion", "LEFT", 2, 70.0, 82.0, 90.0,
                    100, 90, "COMPLETE",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()

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
