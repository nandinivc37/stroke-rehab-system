"""Authentication and role authorization API tests."""

from src.api.dependencies import get_current_user
from src.database.models import ApiUser
from src.services import auth_service


def _create_user(api_client, username, password, role):
    session = api_client.app.state.database_session_factory()
    try:
        return auth_service.create_user(session, username, password, role)
    finally:
        session.close()


def test_login_issues_bearer_token(api_client):
    _create_user(api_client, "therapist1", "long-test-password-123", "therapist")

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
    _create_user(api_client, "therapist2", "long-test-password-123", "therapist")

    response = api_client.post(
        "/api/v1/auth/login",
        json={"username": "therapist2", "password": "incorrect-password"},
    )

    assert response.status_code == 401


def test_login_rejects_inactive_account(api_client):
    user = _create_user(api_client, "inactive", "long-test-password-123", "therapist")
    session = api_client.app.state.database_session_factory()
    try:
        record = session.get(ApiUser, user["user_id"])
        record.is_active = 0
        session.commit()
    finally:
        session.close()

    response = api_client.post(
        "/api/v1/auth/login",
        json={"username": "inactive", "password": "long-test-password-123"},
    )
    assert response.status_code == 401


def test_existing_token_is_rejected_after_account_deactivation(api_client):
    user = _create_user(api_client, "deactivated", "long-test-password-123", "therapist")
    token = auth_service.issue_token(user)["access_token"]

    session = api_client.app.state.database_session_factory()
    try:
        record = session.get(ApiUser, user["user_id"])
        record.is_active = 0
        session.commit()
    finally:
        session.close()

    response = api_client.get(
        "/api/v1/patients",
        headers={"Authorization": f"Bearer {token}"},
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


def test_duplicate_usernames_are_case_insensitive(api_client):
    _create_user(api_client, "same-name", "long-test-password-123", "therapist")
    session = api_client.app.state.database_session_factory()
    try:
        from src.services.auth_service import UserConflictError
        import pytest

        with pytest.raises(UserConflictError):
            auth_service.create_user(
                session, "SAME-NAME", "another-long-password-123", "therapist"
            )
    finally:
        session.close()
