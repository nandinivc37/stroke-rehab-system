"""Authorization tests for therapist-to-patient assignments."""

from src.api.dependencies import get_current_user
from src.database.models import ApiUser


def test_therapist_only_sees_assigned_patient(api_client):
    patient_a = api_client.post(
        "/api/v1/patients", json={"name": "Assigned synthetic patient"}
    ).json()
    patient_b = api_client.post(
        "/api/v1/patients", json={"name": "Unassigned synthetic patient"}
    ).json()

    factory = api_client.app.state.database_session_factory
    session = factory()
    try:
        therapist = ApiUser(
            username="assigned-therapist",
            password_hash="test-only",
            role="therapist",
            is_active=1,
        )
        session.add(therapist)
        session.commit()
        session.refresh(therapist)
        therapist_id = therapist.user_id
    finally:
        session.close()

    assignment = api_client.put(
        f"/api/v1/patients/{patient_a['patient_id']}/therapists/{therapist_id}"
    )
    assert assignment.status_code == 200

    api_client.app.dependency_overrides[get_current_user] = lambda: {
        "user_id": therapist_id,
        "username": "assigned-therapist",
        "role": "therapist",
        "is_active": True,
    }
    try:
        listed = api_client.get("/api/v1/patients")
        assert listed.status_code == 200
        assert [p["patient_id"] for p in listed.json()] == [patient_a["patient_id"]]

        allowed = api_client.get(f"/api/v1/patients/{patient_a['patient_id']}")
        denied = api_client.get(f"/api/v1/patients/{patient_b['patient_id']}")
        assert allowed.status_code == 200
        assert denied.status_code == 404

        denied_assessment = api_client.get(
            f"/api/v1/patients/{patient_b['patient_id']}/assessments/current"
        )
        assert denied_assessment.status_code == 404
    finally:
        api_client.app.dependency_overrides[get_current_user] = lambda: {
            "user_id": 1,
            "username": "test-admin",
            "role": "admin",
            "is_active": True,
        }


def test_therapist_cannot_assign_patients(api_client):
    api_client.app.dependency_overrides[get_current_user] = lambda: {
        "user_id": 9,
        "username": "ordinary-therapist",
        "role": "therapist",
        "is_active": True,
    }
    try:
        response = api_client.put("/api/v1/patients/1/therapists/2")
        assert response.status_code == 403
    finally:
        api_client.app.dependency_overrides[get_current_user] = lambda: {
            "user_id": 1,
            "username": "test-admin",
            "role": "admin",
            "is_active": True,
        }
