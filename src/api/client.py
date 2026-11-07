"""Small synchronous client for the NARS desktop application's API mode.

The client is deliberately independent of PyQt and MediaPipe so its HTTP
contract can be tested without a camera or GUI. Keep tokens in memory only.
"""

from urllib.parse import urlparse
from uuid import UUID

import httpx


class APIClientError(RuntimeError):
    """An API request failed or returned an unexpected response."""


class NARSAPIClient:
    def __init__(self, base_url, *, timeout=10.0, transport=None):
        base_url = base_url.rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API base URL must be an absolute HTTP(S) URL.")
        self.base_url = base_url
        self.timeout = timeout
        self._transport = transport
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )
        self._access_token = None

    @property
    def is_authenticated(self):
        return self._access_token is not None

    @property
    def access_token(self):
        """Return the in-memory token for authenticated companion transports."""
        return self._access_token

    def websocket_url(self, path):
        """Build a ws(s) URL from an absolute API path without altering its route."""
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("WebSocket path must start with a slash.")
        scheme = "wss" if self.base_url.startswith("https://") else "ws"
        authority = self.base_url.split("://", 1)[1]
        return f"{scheme}://{authority}{path}"

    def close(self):
        self._client.close()
        self._access_token = None

    def _request(self, method, path, *, authenticated=True, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        if authenticated:
            if not self._access_token:
                raise APIClientError("Connect to NARS API before making this request.")
            headers["Authorization"] = f"Bearer {self._access_token}"
        try:
            response = self._client.request(
                method, path, headers=headers, **kwargs
            )
        except httpx.HTTPError as exc:
            raise APIClientError(f"Could not reach NARS API: {exc}") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except (ValueError, AttributeError):
                detail = response.text
            if response.status_code == 401:
                self._access_token = None
            raise APIClientError(
                f"NARS API returned HTTP {response.status_code}: {detail}"
            )
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise APIClientError("NARS API returned an invalid JSON response.") from exc

    def login(self, username, password):
        if not username or not password:
            raise ValueError("Username and password are required.")
        result = self._request(
            "POST",
            "/api/v1/auth/login",
            authenticated=False,
            json={"username": username, "password": password},
        )
        token = result.get("access_token") if isinstance(result, dict) else None
        if not token:
            raise APIClientError("NARS API login response did not contain an access token.")
        self._access_token = token
        return self._request("GET", "/api/v1/auth/me")

    def current_user(self):
        return self._request("GET", "/api/v1/auth/me")

    def list_patients(self):
        return self._request("GET", "/api/v1/patients")

    def get_patient(self, patient_id):
        return self._request("GET", f"/api/v1/patients/{int(patient_id)}")

    def start_session(self, patient_id):
        return self._request(
            "POST", "/api/v1/sessions", json={"patient_id": int(patient_id)}
        )

    @staticmethod
    def _session_path_id(session_id):
        try:
            return str(UUID(str(session_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Session ID must be a valid UUID.") from exc

    def submit_pose_frame(self, session_id, payload):
        session_id = self._session_path_id(session_id)
        return self._request(
            "POST", f"/api/v1/sessions/{session_id}/frames", json=payload
        )

    def complete_session(self, session_id):
        session_id = self._session_path_id(session_id)
        return self._request(
            "POST", f"/api/v1/sessions/{session_id}/complete"
        )

    def cancel_session(self, session_id):
        session_id = self._session_path_id(session_id)
        return self._request("DELETE", f"/api/v1/sessions/{session_id}")


def pose_frame_payload(results, timestamp_s, image_width, image_height):
    """Convert a MediaPipe Pose result to the API's validated 33-point frame."""
    def landmarks_to_dicts(collection):
        if collection is None:
            return None
        return [
            {
                "x": float(point.x),
                "y": float(point.y),
                "z": float(point.z),
                "visibility": float(getattr(point, "visibility", 1.0)),
                "presence": (
                    float(point.presence)
                    if getattr(point, "presence", None) is not None else None
                ),
            }
            for point in collection.landmark
        ]

    pose = getattr(results, "pose_landmarks", None)
    world = getattr(results, "pose_world_landmarks", None)
    return {
        "timestamp_s": float(timestamp_s),
        "pose_landmarks": landmarks_to_dicts(pose),
        "pose_world_landmarks": landmarks_to_dicts(world),
        "image_width": int(image_width),
        "image_height": int(image_height),
    }
