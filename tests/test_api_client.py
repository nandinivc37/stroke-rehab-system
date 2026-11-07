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


def test_session_operations_reject_invalid_session_id_before_network_call():
    client = make_client(
        lambda request: pytest.fail("Invalid session IDs must not make requests.")
    )
    client._access_token = "test-token"
    with pytest.raises(ValueError, match="valid UUID"):
        client.submit_pose_frame("../other-resource", {})
    with pytest.raises(ValueError, match="valid UUID"):
        client.complete_session("not-a-uuid")
    with pytest.raises(ValueError, match="valid UUID"):
        client.cancel_session("not-a-uuid")
    client.close()


def test_session_methods_use_expected_http_contract():
    session_id = "12345678-1234-5678-1234-567812345678"
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/frames"):
            return httpx.Response(200, json={"status": "OK", "reps": 1})
        if request.method == "POST":
            return httpx.Response(200, json={"session_id": 22, "status": "COMPLETE"})
        return httpx.Response(200, json={"session_id": session_id, "status": "CANCELLED"})

    client = make_client(handler)
    client._access_token = "test-token"
    assert client.submit_pose_frame(session_id, {"timestamp_s": 1})["reps"] == 1
    assert client.complete_session(session_id)["status"] == "COMPLETE"
    assert client.cancel_session(session_id)["status"] == "CANCELLED"
    assert seen == [
        ("POST", f"/api/v1/sessions/{session_id}/frames"),
        ("POST", f"/api/v1/sessions/{session_id}/complete"),
        ("DELETE", f"/api/v1/sessions/{session_id}"),
    ]
    client.close()

def test_remote_prescription_and_affected_area_requests():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.url.params))
        if request.url.path.endswith("/affected-areas"):
            return httpx.Response(200, json=[{"side": "LEFT"}])
        return httpx.Response(200, json=[{
            "recommendation_id": 4,
            "exercise_id": "shoulder_flexion",
            "status": "APPROVED",
        }])

    client = make_client(handler)
    client._access_token = "test-token"

    areas = client.list_affected_areas(12)
    recommendations = client.list_recommendations(
        12, status_filter="APPROVED"
    )

    assert areas[0]["side"] == "LEFT"
    assert recommendations[0]["status"] == "APPROVED"
    assert seen[0][1] == "/api/v1/patients/12/affected-areas"
    assert seen[1][1] == "/api/v1/patients/12/recommendations"
    assert seen[1][2]["status_filter"] == "APPROVED"

    with pytest.raises(ValueError, match="Unsupported recommendation"):
        client.list_recommendations(12, status_filter="UNKNOWN")
    client.close()

def test_adaptation_evaluation_uses_saved_remote_session_id():
    session_id = 321
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"decision": "MAINTAIN"})

    client = make_client(handler)
    client._access_token = "test-token"
    result = client.evaluate_adaptation(12, session_id)

    assert result["decision"] == "MAINTAIN"
    assert seen == [(
        "POST",
        "/api/v1/patients/12/sessions/321/adaptation/evaluate",
    )]
    client.close()
