import httpx
import pytest

from src.api.client import NARSAPIClient
from src.api.remote_session import RemoteRehabSession, RemoteSessionError


SESSION_ID = "12345678-1234-5678-1234-567812345678"


class FakeStream:
    def __init__(self, url, token, **kwargs):
        self.url = url
        self.token = token
        self.on_analysis = kwargs.get("on_analysis")
        self.on_error = kwargs.get("on_error")
        self.frames = []
        self.flush_called = False
        self.stop_called = False
        self.is_ready = True
        self.error = None

    def start(self):
        return self

    def submit_frame(self, payload):
        self.frames.append(payload)

    def flush(self, timeout=None):
        self.flush_called = True

    def stop(self):
        self.stop_called = True


def make_api_client():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/auth/me"):
            return httpx.Response(200, json={"user_id": 7, "role": "therapist"})
        if request.url.path.endswith("/sessions") and request.method == "POST":
            return httpx.Response(201, json={
                "session_id": SESSION_ID,
                "patient_id": 12,
                "status": "ACTIVE",
            })
        if request.url.path.endswith("/complete"):
            return httpx.Response(200, json={
                "session_id": 91, "patient_id": 12, "status": "COMPLETE",
                "summary": {"frames_total": 1},
            })
        if request.url.path.endswith("/sessions/" + SESSION_ID):
            return httpx.Response(200, json={
                "session_id": SESSION_ID, "status": "CANCELLED"
            })
        return httpx.Response(404, json={"detail": "Not found"})

    client = NARSAPIClient(
        "https://nars.example",
        transport=httpx.MockTransport(handler),
    )
    client._access_token = "in-memory-token"
    return client, calls


def test_remote_session_starts_streams_flushes_and_completes():
    api, calls = make_api_client()
    streams = []
    coordinator = RemoteRehabSession(
        api, stream_factory=lambda *args, **kwargs: streams.append(
            FakeStream(*args, **kwargs)
        ) or streams[-1],
    )

    started = coordinator.start(12)
    coordinator.submit_frame({"timestamp_s": 1.0, "pose_landmarks": None})
    completed = coordinator.complete()

    assert started["session_id"] == SESSION_ID
    assert coordinator.state == "COMPLETED"
    assert coordinator.frames_submitted == 1
    assert streams[0].url == (
        f"wss://nars.example/api/v1/sessions/{SESSION_ID}/stream"
    )
    assert streams[0].token == "in-memory-token"
    assert streams[0].frames == [
        {"timestamp_s": 1.0, "pose_landmarks": None}
    ]
    assert streams[0].flush_called
    assert streams[0].stop_called
    assert completed["status"] == "COMPLETE"
    assert ("POST", f"/api/v1/sessions/{SESSION_ID}/complete") in calls
    api.close()


def test_remote_session_requires_login():
    api, _ = make_api_client()
    api._access_token = None
    coordinator = RemoteRehabSession(api, stream_factory=FakeStream)

    with pytest.raises(RemoteSessionError, match="Log in"):
        coordinator.start(12)
    assert coordinator.state == "IDLE"
    api.close()


def test_remote_session_cancels_backend_if_stream_start_fails():
    api, calls = make_api_client()

    class FailingStream(FakeStream):
        def start(self):
            raise RuntimeError("socket unavailable")

    coordinator = RemoteRehabSession(api, stream_factory=FailingStream)
    with pytest.raises(RemoteSessionError, match="Could not start"):
        coordinator.start(12)

    assert coordinator.state == "FAILED"
    assert ("DELETE", f"/api/v1/sessions/{SESSION_ID}") in calls
    api.close()


def test_remote_session_does_not_complete_without_frames():
    api, _ = make_api_client()
    coordinator = RemoteRehabSession(api, stream_factory=FakeStream)
    coordinator.start(12)

    with pytest.raises(RemoteSessionError, match="No pose frames"):
        coordinator.complete()

    coordinator.cancel()
    assert coordinator.state == "CANCELLED"
    api.close()
