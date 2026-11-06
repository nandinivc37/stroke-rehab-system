from types import SimpleNamespace

import httpx
import pytest

from src.api.client import APIClientError, NARSAPIClient, pose_frame_payload


def make_client(handler):
    return NARSAPIClient(
        "http://nars.test",
        transport=httpx.MockTransport(handler),
    )


def test_login_stores_token_and_returns_current_user():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/auth/login"):
            return httpx.Response(200, json={"access_token": "signed-token"})
        return httpx.Response(200, json={
            "user_id": 7, "username": "therapist", "role": "therapist"
        })

    client = make_client(handler)
    user = client.login("therapist", "correct-password")

    assert user["user_id"] == 7
    assert client.is_authenticated
    assert calls[1].headers["Authorization"] == "Bearer signed-token"
    client.close()


def test_protected_requests_require_login():
    client = make_client(lambda request: httpx.Response(200, json=[]))
    with pytest.raises(APIClientError, match="Connect to NARS API"):
        client.list_patients()
    client.close()


def test_patient_list_uses_authenticated_endpoint():
    def handler(request):
        assert request.url.path == "/api/v1/patients"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(200, json=[{"patient_id": 12, "name": "Test"}])

    client = make_client(handler)
    client._access_token = "test-token"
    assert client.list_patients()[0]["patient_id"] == 12
    client.close()


def test_unauthorized_response_clears_token():
    client = make_client(
        lambda request: httpx.Response(401, json={"detail": "Token expired"})
    )
    client._access_token = "expired-token"
    with pytest.raises(APIClientError, match="HTTP 401"):
        client.list_patients()
    assert not client.is_authenticated
    client.close()


def test_pose_frame_payload_preserves_33_landmarks_and_dimensions():
    points = [
        SimpleNamespace(x=i / 33, y=0.5, z=-0.1, visibility=0.9, presence=0.8)
        for i in range(33)
    ]
    results = SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=points),
        pose_world_landmarks=SimpleNamespace(landmark=points),
    )

    payload = pose_frame_payload(results, 1.25, 1280, 720)

    assert payload["timestamp_s"] == 1.25
    assert len(payload["pose_landmarks"]) == 33
    assert len(payload["pose_world_landmarks"]) == 33
    assert payload["pose_landmarks"][10]["visibility"] == 0.9
    assert payload["image_width"] == 1280
    assert payload["image_height"] == 720


def test_pose_frame_payload_supports_no_detected_pose():
    payload = pose_frame_payload(
        SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None),
        2.0, 640, 480,
    )
    assert payload["pose_landmarks"] is None
    assert payload["pose_world_landmarks"] is None
