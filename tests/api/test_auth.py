"""Authentication and role authorization API tests."""

from src.api.dependencies import get_current_user
from src.services import auth_service


def test_login_issues_bearer_token(api_client):
    auth_service.create_user("therapist1", "long-test-password-123", "therapist")

    response = api_client.post(
        "/api/v1/auth/login",
        json={"username": "therapist1", "password": "long-test-password-123"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["expires_in"] > 0
    assert payload["access_token"]


def test_login_rejects_invalid_password(api_client):
    auth_service.create_user("therapist2", "long-test-password-123", "therapist")

    response = api_client.post(
        "/api/v1/auth/login",
        json={"username": "therapist2", "password": "incorrect-password"},
    )

    assert response.status_code == 401


def test_patient_routes_reject_requests_without_bearer_token(api_client):
    api_client.app.dependency_overrides.pop(get_current_user, None)
    response = api_client.get("/api/v1/patients")
    assert response.status_code == 401


def test_therapist_cannot_provision_users(api_client):
    response = api_client.post(
        "/api/v1/auth/users",
        json={
            "username": "another-therapist",
            "password": "long-test-password-123",
            "role": "therapist",
        },
    )
    assert response.status_code == 403


def test_admin_can_provision_therapist(api_client):
    api_client.app.dependency_overrides[get_current_user] = lambda: {
        "user_id": 2,
        "username": "test-admin",
        "role": "admin",
        "is_active": True,
    }
    response = api_client.post(
        "/api/v1/auth/users",
        json={
            "username": "created-therapist",
            "password": "long-test-password-123",
            "role": "therapist",
        },
    )
    assert response.status_code == 201
    assert response.json()["role"] == "therapist"
    assert "password" not in response.json()
    assert "password_hash" not in response.json()
