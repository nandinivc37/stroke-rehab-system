import json
import threading

import pytest

from src.api.pose_stream import PoseStreamClient, PoseStreamError


class FakeConnection:
    def __init__(self):
        self.sent = []
        self._responses = ["{\"type\":\"authenticated\",\"user_id\":7}"]
        self.frame_received = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def send(self, message):
        self.sent.append(json.loads(message))
        if self.sent[-1].get("type") == "frame":
            self._responses.append(
                "{\"type\":\"analysis\",\"data\":{\"reps\":1}}"
            )
            self.frame_received.set()

    def recv(self, timeout=None):
        if self._responses:
            return self._responses.pop(0)
        raise TimeoutError("No fake response available.")


def test_pose_stream_authenticates_and_delivers_analysis():
    connection = FakeConnection()
    analyses = []
    received = threading.Event()

    def on_analysis(data):
        analyses.append(data)
        received.set()

    client = PoseStreamClient(
        "ws://nars.test/api/v1/sessions/123/stream",
        "signed-token",
        on_analysis=on_analysis,
        connection_factory=lambda *_args, **_kwargs: connection,
    )
    client.start()
    client.submit_frame({"timestamp_s": 1.0, "pose_landmarks": None})

    assert received.wait(2)
    assert analyses == [{"reps": 1}]
    assert connection.sent[0] == {
        "type": "authenticate",
        "access_token": "signed-token",
    }
    assert connection.sent[1] == {
        "type": "frame",
        "frame": {"timestamp_s": 1.0, "pose_landmarks": None},
    }
    client.stop()


def test_pose_stream_rejects_invalid_url_and_missing_token():
    with pytest.raises(ValueError, match="WebSocket URL"):
        PoseStreamClient("http://nars.test/stream", "token")
    with pytest.raises(ValueError, match="access token"):
        PoseStreamClient("ws://nars.test/stream", "")
