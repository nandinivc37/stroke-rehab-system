"""Coordinates an authenticated desktop-to-API rehabilitation session.

This class deliberately contains no Qt dependencies. The GUI owns the
camera/MediaPipe widget and passes already-computed landmark payloads here.
"""

import threading

from src.api.client import APIClientError, NARSAPIClient
from src.api.pose_stream import PoseStreamClient, PoseStreamError


class RemoteSessionError(RuntimeError):
    """A remote rehabilitation session could not be started or maintained."""


class RemoteRehabSession:
    def __init__(
        self,
        api_client,
        *,
        on_analysis=None,
        on_error=None,
        stream_factory=PoseStreamClient,
    ):
        if not isinstance(api_client, NARSAPIClient):
            raise TypeError("api_client must be a NARSAPIClient instance.")
        self.api_client = api_client
        self.on_analysis = on_analysis
        self.on_error = on_error
        self.stream_factory = stream_factory
        self.session = None
        self.stream = None
        self._lock = threading.RLock()
        self._state = "IDLE"
        self._frames_submitted = 0
        self._last_error = None

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def session_id(self):
        return self.session.get("session_id") if self.session else None

    @property
    def frames_submitted(self):
        with self._lock:
            return self._frames_submitted

    @property
    def last_error(self):
        return self._last_error

    def start(self, patient_id):
        with self._lock:
            if self._state != "IDLE":
                raise RemoteSessionError(
                    f"Cannot start a remote session while state is {self._state}."
                )
            if not self.api_client.is_authenticated:
                raise RemoteSessionError("Log in to the NARS API before starting a session.")
            self._state = "STARTING"

        created = None
        try:
            created = self.api_client.start_session(patient_id)
            session_id = created.get("session_id")
            if not session_id:
                raise RemoteSessionError("API did not return a session identifier.")

            stream_path = f"/api/v1/sessions/{session_id}/stream"
            stream = self.stream_factory(
                self.api_client.websocket_url(stream_path),
                self.api_client.access_token,
                on_analysis=self.on_analysis,
                on_error=self._stream_failed,
            )
            stream.start()
            with self._lock:
                self.session = created
                self.stream = stream
                self._state = "ACTIVE"
                self._frames_submitted = 0
            return created
        except Exception as exc:
            if created and created.get("session_id"):
                try:
                    self.api_client.cancel_session(created["session_id"])
                except Exception:
                    # Preserve the initiating failure; a durable active
                    # session will be marked interrupted by idle cleanup.
                    pass
            with self._lock:
                self._state = "FAILED"
                self._last_error = exc
            raise RemoteSessionError(f"Could not start remote session: {exc}") from exc

    def submit_frame(self, payload):
        with self._lock:
            if self._state != "ACTIVE" or self.stream is None:
                raise RemoteSessionError("No active remote rehabilitation session.")
            stream = self.stream
        if not stream.is_ready:
            error = stream.error or PoseStreamError("Pose stream is no longer connected.")
            self._stream_failed(error)
            raise RemoteSessionError(str(error)) from error
        try:
            stream.submit_frame(payload)
        except Exception as exc:
            self._stream_failed(exc)
            raise RemoteSessionError(f"Could not submit pose frame: {exc}") from exc
        with self._lock:
            self._frames_submitted += 1

    def complete(self):
        with self._lock:
            if self._state != "ACTIVE":
                raise RemoteSessionError(
                    f"Cannot complete a remote session while state is {self._state}."
                )
            if self._frames_submitted == 0:
                raise RemoteSessionError(
                    "No pose frames were submitted; the session cannot be completed."
                )
            self._state = "COMPLETING"
            stream = self.stream
            session_id = self.session["session_id"]

        if stream is not None:
            stream.stop()
        try:
            result = self.api_client.complete_session(session_id)
        except APIClientError as exc:
            with self._lock:
                self._state = "FAILED"
                self._last_error = exc
            raise RemoteSessionError(
                f"Pose stream stopped, but the API could not save the session: {exc}"
            ) from exc

        with self._lock:
            self._state = "COMPLETED"
            self.stream = None
        return result

    def cancel(self):
        with self._lock:
            if self._state not in {"ACTIVE", "FAILED"} or not self.session:
                if self._state in {"IDLE", "CANCELLED"}:
                    return None
                raise RemoteSessionError(
                    f"Cannot cancel a remote session while state is {self._state}."
                )
            self._state = "CANCELLING"
            stream = self.stream
            session_id = self.session["session_id"]

        if stream is not None:
            stream.stop()
        try:
            result = self.api_client.cancel_session(session_id)
        except APIClientError as exc:
            with self._lock:
                self._state = "FAILED"
                self._last_error = exc
            raise RemoteSessionError(f"Could not cancel remote session: {exc}") from exc

        with self._lock:
            self._state = "CANCELLED"
            self.stream = None
        return result

    def _stream_failed(self, error):
        with self._lock:
            if self._state not in {"ACTIVE", "STARTING"}:
                return
            self._last_error = error
            self._state = "FAILED"
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception:
                pass
