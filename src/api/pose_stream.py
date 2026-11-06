"""Background WebSocket transport for live NARS pose-landmark sessions.

Camera capture and MediaPipe inference stay on the desktop. This transport
ships already-computed landmarks to the API without blocking the Qt capture
thread. The bounded queue deliberately drops stale queued frames under
network pressure; it never reorders frames that are sent.
"""

import json
import queue
import threading
from urllib.parse import urlparse


class PoseStreamError(RuntimeError):
    """The live pose stream could not be established or maintained."""


class PoseStreamClient:
    def __init__(
        self,
        websocket_url,
        access_token,
        *,
        on_analysis=None,
        on_error=None,
        queue_size=2,
        connect_timeout=10.0,
        response_timeout=10.0,
        connection_factory=None,
    ):
        parsed = urlparse(websocket_url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ValueError("WebSocket URL must be an absolute ws:// or wss:// URL.")
        if not access_token:
            raise ValueError("An access token is required for the pose stream.")
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1.")

        self.websocket_url = websocket_url
        self.access_token = access_token
        self.on_analysis = on_analysis
        self.on_error = on_error
        self.connect_timeout = connect_timeout
        self.response_timeout = response_timeout
        self.connection_factory = connection_factory
        self._frames = queue.Queue(maxsize=queue_size)
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._thread = None
        self._error = None
        self.dropped_frames = 0

    @property
    def is_ready(self):
        return self._ready.is_set() and self._error is None and not self._stopped.is_set()

    @property
    def error(self):
        return self._error

    def start(self):
        if self._thread is not None:
            raise PoseStreamError("Pose stream has already been started.")
        self._thread = threading.Thread(
            target=self._run,
            name="nars-pose-stream",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(self.connect_timeout + 1):
            self.stop()
            raise PoseStreamError("Timed out while authenticating the pose stream.")
        if self._error is not None:
            self.stop()
            raise PoseStreamError(str(self._error)) from self._error
        return self

    def submit_frame(self, frame_payload):
        """Queue one validated landmark payload without waiting for the network."""
        if not self.is_ready:
            raise PoseStreamError("Pose stream is not connected.")
        try:
            encoded = json.dumps({"type": "frame", "frame": frame_payload})
        except (TypeError, ValueError) as exc:
            raise ValueError("Frame payload must be JSON serializable.") from exc

        try:
            self._frames.put_nowait(encoded)
        except queue.Full:
            # Prefer the newest camera observation over stale queued frames.
            try:
                self._frames.get_nowait()
                self._frames.task_done()
            except queue.Empty:
                pass
            self.dropped_frames += 1
            self._frames.put_nowait(encoded)

    def stop(self, join_timeout=None):
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            try:
                self._frames.put_nowait(None)
            except queue.Full:
                try:
                    self._frames.get_nowait()
                    self._frames.task_done()
                except queue.Empty:
                    pass
                try:
                    self._frames.put_nowait(None)
                except queue.Full:
                    pass
            self._thread.join(
                self.response_timeout + 1 if join_timeout is None else join_timeout
            )
        self._ready.clear()

    def _notify_error(self, error):
        self._error = error
        self._ready.set()
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception:
                # A presentation callback must not obscure the transport error.
                pass

    def _run(self):
        try:
            factory = self.connection_factory
            if factory is None:
                from websockets.sync.client import connect
                factory = connect

            with factory(
                self.websocket_url,
                open_timeout=self.connect_timeout,
                close_timeout=2,
                max_size=2**20,
            ) as connection:
                connection.send(json.dumps({
                    "type": "authenticate",
                    "access_token": self.access_token,
                }))
                authentication = json.loads(
                    connection.recv(timeout=self.connect_timeout)
                )
                if authentication.get("type") != "authenticated":
                    raise PoseStreamError(
                        authentication.get("detail", "Pose stream authentication failed.")
                    )
                self._ready.set()

                while not self._stopped.is_set():
                    try:
                        message = self._frames.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if message is None:
                        self._frames.task_done()
                        break
                    try:
                        connection.send(message)
                        response = json.loads(
                            connection.recv(timeout=self.response_timeout)
                        )
                        if response.get("type") == "error":
                            raise PoseStreamError(
                                response.get("detail", "The API rejected a pose frame.")
                            )
                        if response.get("type") != "analysis":
                            raise PoseStreamError(
                                "The API returned an unexpected pose-stream message."
                            )
                        if self.on_analysis is not None:
                            self.on_analysis(response.get("data", {}))
                    finally:
                        self._frames.task_done()
        except Exception as exc:
            if not self._stopped.is_set():
                self._notify_error(exc)
        finally:
            self._ready.clear()
