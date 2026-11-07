import logging
import os
import statistics
import sys
import time
from collections import deque

import cv2
import mediapipe as mp

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.adaptation.policy import PolicyConfig
from src.compensation_detection.geometric_detector import (
    GeometricCompensationDetector,
    feedback as compensation_feedback,
)
from src.compensation_detection.geometric_measures import (
    ExperimentalPostureMonitor,
)
from src.exercises.analyzer import (
    REP_COUNTED,
    ExerciseAnalyzer,
    STATUS_IMPLAUSIBLE,
    STATUS_NO_POSE,
    STATUS_NOT_VISIBLE,
    STATUS_OK,
    STATUS_UNMEASURABLE,
)
from src.exercises.angles import PLANE_CALIBRATING, PLANE_DEPTH_AMBIGUOUS
from src.exercises.legacy import LEGACY_SHOULDER_RAISE_2D
from src.pose_estimation.config import pose_settings
from src.validation.landmark_recorder import LandmarkRecorder
from src.api.client import pose_frame_payload

logger = logging.getLogger(__name__)

# The Kinect-trained BiLSTM compensation model is intentionally NOT used
# here: its features are incompatible with MediaPipe landmarks (see
# README, "Compensation detection status"). Live compensation feedback
# comes from the rule-based geometric detector (provisional thresholds,
# not clinically validated, never used for adaptation).

# Stop analysing after this many consecutive frames fail, rather than
# silently continuing with a broken pipeline.
MAX_CONSECUTIVE_ERRORS = 30
MAX_CONSECUTIVE_READ_FAILURES = 60

# Frames used for the "now" frame rate shown live (about 2-3 s at the
# rates seen on a CPU). The session average is the analyzer's mean_fps,
# the same value the adaptive policy's frame-rate gate checks.
RECENT_FPS_FRAMES = 20

# Shown while the movement plane cannot be verified (front-facing
# shoulder flexion); such repetitions are not counted.
PLANE_HINTS = {
    PLANE_CALIBRATING: "\nRest the arm by your side for a moment to "
                       "calibrate (repetitions are not counted yet).",
    PLANE_DEPTH_AMBIGUOUS: "\nMovement direction unclear: face the "
                           "camera and raise the arm forward.",
}

# Temporary diagnostics (off by default). REHAB_DEBUG_OVERLAY=1 shows the
# repetition decision state under the video and logs it per frame at
# DEBUG level; REHAB_RECORD_LANDMARKS=<file> appends every frame's
# landmarks and decision state to a JSON Lines file for offline replay
# (python -m src.validation.flexion_trace <file>).
DEBUG_OVERLAY_ENV = "REHAB_DEBUG_OVERLAY"
RECORD_LANDMARKS_ENV = "REHAB_RECORD_LANDMARKS"


def _deg(value):
    return "—" if value is None else f"{value:.0f}°"


def format_debug_state(state):
    """Multi-line text of ExerciseAnalyzer.debug_state()."""
    if state["plane_ok"] is None:
        check = state["plane_status"] or "not assessed"
    else:
        check = "OK" if state["plane_ok"] else "OUT OF RANGE"
    expected = (
        "none" if state["expected_plane"] is None
        else f"{state['expected_plane']:g}±{state['plane_tolerance']:g}°"
    )
    last = state["last_rep"]
    distance = state.get("camera_distance")
    camera = "" if distance is None else f" | camera ~{distance:.1f} m"
    return "\n".join([
        f"DEBUG  elevation {_deg(state['angle'])} | target "
        f"{state['target']:g}° | lower below {state['down_threshold']:g}°",
        f"plane {_deg(state['plane'])} (image {_deg(state['image_plane'])},"
        f" world {_deg(state['world_plane'])}) | expected {expected} | "
        f"{check}{camera}",
        f"segment {'ACTIVE' if state['segment_active'] else 'idle'} "
        f"(rest {_deg(state.get('resting_angle'))}; last end: "
        f"{state.get('segment_end_reason') or '—'}) | "
        f"rep stage {state['stage']} | reps {state['reps']}, off-plane "
        f"{state['off_plane_reps']}, unverified "
        f"{state['plane_unverified_reps']}, interrupted "
        f"{state['interrupted_reps']}, bridged {state.get('bridged_gaps', 0)}",
        f"now: {state['pending']}",
        f"last rep: {last['reason'] if last else '—'}",
    ] + _compensation_debug_lines(state.get("compensation")))


def _compensation_debug_lines(diagnostics):
    """Overlay lines for GeometricCompensationDetector.diagnostics()."""
    if not diagnostics:
        return []
    short = {"trunk_lean_image_plane": "lean side",
             "trunk_lean_depth": "lean fwd", "shoulder_hiking": "hiking"}
    parts = []
    for name, p in diagnostics["patterns"].items():
        if p["deviation"] is not None:
            detail = f"{p['deviation']:+.2f}/{p['threshold']:g}"
        else:
            detail = p["reason"] or "-"
        parts.append(f"{short.get(name, name)} {p['state']} ({detail})")
    samples = max((p["baseline_samples"]
                   for p in diagnostics["patterns"].values()), default=0)
    return [
        "posture: baseline "
        + ("ready" if diagnostics["baseline_established"]
           else f"calibrating ({samples} samples)")
        + f" | trunk {_deg(diagnostics['trunk_inclination'])} from vertical",
        " | ".join(parts) if parts else "posture: no data",
    ]


CAMERA_LOST_FEEDBACK = (
    "Posture check (experimental): unable to assess — camera not "
    "delivering frames", "info")

# Session status values stored in sessions.session_status.
SESSION_COMPLETE = "COMPLETE"
SESSION_ANALYSIS_HALTED = "ANALYSIS_HALTED"
SESSION_CAMERA_LOST = "CAMERA_LOST"

FPS_STYLE = """
    background-color: #111827;
    color: {colour};
    font-size: 16px;
    font-weight: 700;
    padding: 6px 10px;
    border-radius: 6px;
"""
FPS_OK_COLOUR = "#ffffff"
FPS_LOW_COLOUR = "#fde047"

COMPENSATION_STYLE = """
    background-color: {background};
    color: {colour};
    font-size: 15px;
    font-weight: 700;
    padding: 6px 10px;
    border-radius: 6px;
"""
# (background, text) per feedback level; all pairs >= 4.5:1 contrast.
COMPENSATION_COLOURS = {
    "alert": ("#fef3c7", "#78350f"),
    "ok": ("#dcfce7", "#14532d"),
    "info": ("#f3f4f6", "#374151"),
}


def _timing_summary(values_s):
    if not values_s:
        return None
    ms = sorted(1000.0 * v for v in values_s)
    return {
        "median_ms": round(statistics.median(ms), 1),
        "p90_ms": round(ms[min(len(ms) - 1, int(0.9 * len(ms)))], 1),
        "mean_ms": round(sum(ms) / len(ms), 1),
        "n": len(ms),
    }


class CaptureWorker(QThread):
    """
    Runs camera capture, pose estimation and analysis off the GUI
    thread. Only the worker touches the camera, the pose model and the
    analyzer while it runs; results reach the GUI through the queued
    frame_ready signal.
    """

    frame_ready = pyqtSignal(object)

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        widget = self.widget
        while not self._stop_requested and not widget.analysis_halted:
            output = widget._step()
            if output is None:
                self.msleep(5)
                continue
            # Drop the picture (not the analysis) if the GUI has not
            # shown the previous one yet, so frames never queue up.
            if widget._display_pending:
                output["display"] = None
            else:
                widget._display_pending = output["display"] is not None
            self.frame_ready.emit(output)


class LivePoseWidget(QWidget):

    pose_frame_ready = pyqtSignal(object)

    def __init__(
        self,
        exercise=None,
        affected_side="LEFT",
        parameters=None,
        parent=None,
        threaded=True,
    ):
        super().__init__(parent)

        self.exercise = exercise or LEGACY_SHOULDER_RAISE_2D
        self.affected_side = affected_side.upper()
        self.threaded = threaded

        self.analyzer = ExerciseAnalyzer(
            self.exercise,
            self.affected_side,
            parameters,
        )

        # Experimental, unvalidated posture measurements: recorded for
        # therapist review only, never shown on the patient overlay.
        self.posture_monitor = ExperimentalPostureMonitor(
            self.affected_side
        )

        # Rule-based compensation detection on the same landmarks.
        self.compensation = GeometricCompensationDetector(
            self.affected_side,
            getattr(self.exercise, "exercise_id", None),
        )

        self.camera = None
        self.pose = None
        self._compensation_feedback = None
        self.worker = None
        self.pose_config = None
        self._display_pending = False

        self.consecutive_errors = 0
        self.consecutive_read_failures = 0
        self.read_failures_total = 0
        self.camera_lost = False
        self.last_error = None
        self.analysis_halted = False
        self.frame_size = None
        self._timings = {"read": [], "inference": [], "analysis": []}
        self.debug_overlay = os.environ.get(DEBUG_OVERLAY_ENV) == "1"
        self.recorder = None
        self._debug_text = None
        self._last_pose_payload = None

        self.video_label = QLabel()
        self.video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.video_label.setMinimumSize(500, 400)
        self.video_label.setStyleSheet("""
            background-color: #111827;
            border-radius: 10px;
        """)

        self.status_label = QLabel(
            "Camera not started"
        )
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("""
            color: #374151;
            font-size: 14px;
            font-weight: 600;
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Measured processing frame rate, shown while the exercise runs.
        self.min_fps = PolicyConfig().min_fps
        self.frame_times = deque(maxlen=RECENT_FPS_FRAMES)
        self.fps_label = QLabel("Frame rate: —")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label.setWordWrap(True)
        self.fps_label.setStyleSheet(
            FPS_STYLE.format(colour=FPS_OK_COLOUR)
        )

        self.compensation_label = QLabel(
            "Posture check (experimental): waiting for camera"
        )
        self.compensation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compensation_label.setWordWrap(True)
        self.compensation_label.setToolTip(
            "Rule-based check of trunk lean and shoulder hiking relative to "
            "your resting posture. Provisional thresholds; not clinically "
            "validated; does not change exercise difficulty."
        )
        self._set_compensation_label(
            "Posture check (experimental): waiting for camera", "info")

        self.debug_label = QLabel()
        self.debug_label.setWordWrap(True)
        self.debug_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #111827; "
            "background-color: #fef9c3; padding: 4px;"
        )
        self.debug_label.setVisible(self.debug_overlay)

        layout.addWidget(self.video_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.compensation_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.debug_label)

        # Unthreaded fallback (threaded=False).
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    # -----------------------------------------------------
    # Camera lifecycle
    # -----------------------------------------------------

    def start(self):

        if self.camera is not None:
            return

        if sys.platform.startswith("linux"):
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        elif sys.platform == "win32":
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.camera = cv2.VideoCapture(0)

        self.camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        self.camera.set(
            cv2.CAP_PROP_FPS,
            30
        )

        # Keep only the newest frame where the backend supports it, so
        # analysis is not fed stale buffered frames when it is slower
        # than the camera.
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.camera.isOpened():
            self.status_label.setText(
                "Could not open camera"
            )
            self.camera = None
            return

        self.pose_config = pose_settings()
        self.pose = mp.solutions.pose.Pose(**self.pose_config)

        self._reset_session_state()
        self._start_recorder()

        self.status_label.setText(
            f"Monitoring {self.affected_side.lower()} arm"
        )

        if self.threaded:
            self.worker = CaptureWorker(self)
            self.worker.frame_ready.connect(self._present)
            self.worker.start()
        else:
            self.timer.start(30)

    def _reset_session_state(self):
        self.analyzer.reset()
        self.posture_monitor.reset()
        self.compensation.reset()
        self.frame_times.clear()
        self.consecutive_errors = 0
        self.consecutive_read_failures = 0
        self.read_failures_total = 0
        self.camera_lost = False
        self.last_error = None
        self.analysis_halted = False
        self._display_pending = False
        self._timings = {"read": [], "inference": [], "analysis": []}
        self._last_pose_payload = None

    def _start_recorder(self):
        path = os.environ.get(RECORD_LANDMARKS_ENV)
        if not path or self.recorder is not None:
            return
        try:
            self.recorder = LandmarkRecorder(path, {
                "exercise_id": self.exercise.exercise_id,
                "side": self.affected_side,
                "parameters": dict(self.analyzer.parameters),
                "pose_settings": self.pose_config,
            })
            logger.warning("Recording landmarks to %s", path)
        except OSError:
            logger.exception("Could not open landmark recording %s", path)

    def _diagnose(self, timestamp, results):
        """Debug overlay text and landmark recording; never raises."""
        if not (self.debug_overlay or self.recorder):
            return
        try:
            state = self.analyzer.debug_state()
            state["compensation"] = self.compensation.diagnostics()
            state["compensation"]["shown"] = (
                None if self._compensation_feedback is None
                else self._compensation_feedback[0])
            if self.debug_overlay:
                self._debug_text = format_debug_state(state)
                logger.debug("Repetition state: %s", state)
            if self.recorder is not None:
                self.recorder.record(timestamp, self.frame_size, results,
                                     state)
        except Exception:
            logger.exception("Diagnostics failed")

    def _stop_worker(self):
        """Stop the capture thread and wait for it (idempotent)."""

        worker = self.worker
        if worker is None:
            return
        worker.request_stop()
        if not worker.wait(3000):
            # A blocked camera read: releasing the device unblocks it.
            logger.warning("Capture thread did not stop; releasing camera")
            if self.camera is not None:
                self.camera.release()
            worker.wait(3000)
        self.worker = None

    # -----------------------------------------------------
    # Per-frame processing
    # -----------------------------------------------------

    def update_frame(self):
        """Process one frame and show it (unthreaded path and tests)."""
        output = self._step()
        if output is not None:
            self._present(output)

    def _timestamp(self):
        stamp = getattr(self.camera, "timestamp", None)
        value = stamp() if callable(stamp) else None
        return value if value is not None else time.monotonic()

    def _step(self):
        """
        Read, analyse and draw one frame without touching any Qt widget
        (safe on the capture thread). Returns the output to present, or
        None when no frame was read and nothing changed.
        """

        if self.camera is None or self.analysis_halted:
            return None

        t0 = time.perf_counter()
        success, frame = self.camera.read()
        read_time = time.perf_counter() - t0

        if not success:
            self.consecutive_read_failures += 1
            self.read_failures_total += 1

            if (
                self.consecutive_read_failures
                == MAX_CONSECUTIVE_READ_FAILURES
            ):
                self.camera_lost = True
                # Clear any previous posture verdict.
                self._compensation_feedback = CAMERA_LOST_FEEDBACK
                return {
                    "display": None,
                    "status": (
                        "Camera is not delivering frames. "
                        "Check the connection and restart the session."
                    ),
                    "fps": None,
                    "compensation": self._compensation_feedback,
                }
            return None

        timestamp = self._timestamp()
        self.consecutive_read_failures = 0
        self.frame_times.append(timestamp)
        self._timings["read"].append(read_time)
        self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        self.analyzer.set_image_size(*self.frame_size)

        # A Qt slot must never raise: PyQt6 aborts the whole application
        # on an unhandled exception. Errors are logged, counted in the
        # session summary, and shown; no substitute values are produced.
        self._last_pose_payload = None
        try:
            display, status = self._process_frame(frame, timestamp)
            self.consecutive_errors = 0

        except Exception as error:
            logger.exception("Frame processing failed")

            self.analyzer.record_error(timestamp)
            self.consecutive_errors += 1
            self.last_error = f"{type(error).__name__}: {error}"
            display = None
            # Never leave a stale posture verdict on screen.
            self._compensation_feedback = (
                "Posture check (experimental): unable to assess "
                "(frame processing error)", "info")

            if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self.analysis_halted = True
                status = (
                    "Analysis stopped after repeated processing errors "
                    f"({self.last_error}). End the session; "
                    "data recorded so far is kept."
                )
            else:
                status = f"Frame processing error: {self.last_error}"

        return {"display": display, "status": status, "fps": self._fps(),
                "compensation": self._compensation_feedback,
                "debug": self._debug_text,
                "pose_frame": self._last_pose_payload}

    def recent_fps(self):
        """Processed frames per second over the last few frames."""
        if len(self.frame_times) < 2:
            return None
        span = self.frame_times[-1] - self.frame_times[0]
        if span <= 0:
            return None
        return (len(self.frame_times) - 1) / span

    def _fps(self):
        """(text, colour) for the frame-rate label."""
        recent = self.recent_fps()
        average = self.analyzer.mean_fps

        if recent is None or average is None:
            return "Frame rate: measuring…", FPS_OK_COLOUR

        text = (
            f"Frame rate: {recent:.1f} fps now · "
            f"{average:.1f} fps session average"
        )
        if average < self.min_fps:
            text += (
                f" — below {self.min_fps:g} fps: this attempt "
                "cannot be assessed"
            )
            return text, FPS_LOW_COLOUR
        return text, FPS_OK_COLOUR

    def _process_frame(self, frame, timestamp):

        # MediaPipe receives the original (unmirrored) frame.
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        t0 = time.perf_counter()
        results = self.pose.process(rgb)
        t1 = time.perf_counter()
        self._last_pose_payload = pose_frame_payload(
            results, timestamp, frame.shape[1], frame.shape[0]
        )

        analysis = self.analyzer.update(
            results,
            timestamp,
        )

        self.posture_monitor.update(
            results,
            analysis.movement_active,
        )

        self._update_compensation(
            results, analysis.movement_active,
            arm_at_rest=(analysis.status == STATUS_OK
                         and not analysis.movement_active),
        )
        self._diagnose(timestamp, results)

        display = frame.copy()

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                display,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )

        # Mirror for user-facing display BEFORE drawing any text,
        # otherwise the text itself appears mirrored.
        display = cv2.flip(display, 1)

        self._draw_overlay(display, analysis)

        self._timings["inference"].append(t1 - t0)
        self._timings["analysis"].append(time.perf_counter() - t1)

        return display, self._status_text()

    def _update_compensation(self, results, movement_active,
                             arm_at_rest=None):
        """
        Compensation check for this frame. Its failure must not stop
        repetition counting or DTW, so errors only disable the check.
        It uses only the attempt signal (angle-based segmentation), never
        repetition counting or plane validation.
        """
        try:
            frame = self.compensation.update(results, movement_active,
                                             arm_at_rest)
            self._compensation_feedback = compensation_feedback(
                frame, self.compensation.config, movement_active,
                self.compensation.baseline_established(),
            )
        except Exception as error:
            logger.exception("Compensation check failed")
            self._compensation_feedback = (
                "Posture check (experimental): unable to assess "
                f"(error: {type(error).__name__})", "info")

    def _set_compensation_label(self, text, level):
        background, colour = COMPENSATION_COLOURS.get(
            level, COMPENSATION_COLOURS["info"])
        self.compensation_label.setText(text)
        self.compensation_label.setStyleSheet(COMPENSATION_STYLE.format(
            background=background, colour=colour))

    def _present(self, output):
        """GUI-thread half: show a step's output."""

        if self.analysis_halted:
            self.timer.stop()

        if output.get("display") is not None:
            self._show(output["display"])
        if output.get("pose_frame") is not None:
            self.pose_frame_ready.emit(output["pose_frame"])
        self._display_pending = False

        if output.get("status"):
            self.status_label.setText(output["status"])

        if output.get("fps") is not None:
            text, colour = output["fps"]
            self.fps_label.setText(text)
            self.fps_label.setStyleSheet(FPS_STYLE.format(colour=colour))

        if output.get("compensation") is not None:
            self._set_compensation_label(*output["compensation"])

        if output.get("debug") is not None:
            self.debug_label.setText(output["debug"])

    def _reps_text(self):
        target = self.analyzer.parameters.get("target_reps")
        if target:
            return f"Reps: {self.analyzer.reps} / {int(target)}"
        return f"Reps: {self.analyzer.reps}"

    def _draw_overlay(self, display, analysis):

        if analysis.status == STATUS_OK:
            lines = [
                (
                    f"{self.exercise.angle_label}: "
                    f"{analysis.angle:.1f}",
                    (255, 255, 255),
                    0.9,
                ),
                (self._reps_text(), (255, 255, 255), 0.9),
            ]
        elif analysis.status == STATUS_NOT_VISIBLE:
            lines = [("Keep affected arm visible", (0, 200, 255), 0.8)]
        elif analysis.status in (STATUS_UNMEASURABLE, STATUS_IMPLAUSIBLE):
            lines = [("Cannot measure this frame", (0, 200, 255), 0.8)]
        elif analysis.status == STATUS_NO_POSE:
            lines = [("Pose not detected", (0, 200, 255), 0.8)]
        else:
            lines = []

        y = 45
        for text, colour, scale in lines:
            cv2.putText(
                display,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                colour,
                2,
            )
            y += 40

    def _show(self, display):

        rgb_display = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = rgb_display.shape

        bytes_per_line = channels * width

        image = QImage(
            rgb_display.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image)

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _status_text(self):
        # Never show the last measured angle for a frame that was not
        # measured: it makes an untracked arm look tracked.
        if self.analyzer.last_status == STATUS_OK:
            angle = f"{self.analyzer.current_angle:.1f}°"
        else:
            angle = "not tracked"
        fraction = self.analyzer.measured_fraction
        tracked = (
            f"    Arm tracked: {100 * fraction:.0f}% of frames"
            if fraction is not None else ""
        )
        return (
            f"{self._reps_text()}    "
            f"{self.exercise.angle_label}: {angle}{tracked}"
            f"{self._plane_hint()}{self._rep_feedback()}"
        )

    def _rep_feedback(self):
        """Why the current or last repetition does not count, so a
        rejected repetition is never silent."""
        pending = self.analyzer.pending_reason()
        if pending.startswith("will NOT count"):
            return f"\nThis repetition {pending}"
        events = self.analyzer.rep_events
        if events and events[-1]["outcome"] != REP_COUNTED:
            return f"\nLast repetition {events[-1]['reason']}"
        return ""

    def _plane_hint(self):
        """Say why repetitions cannot currently be verified."""
        if self.analyzer.last_status != STATUS_OK:
            return ""
        return PLANE_HINTS.get(self.analyzer.current_plane_status, "")

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    def session_status(self):
        if self.analysis_halted:
            return SESSION_ANALYSIS_HALTED
        if self.camera_lost:
            return SESSION_CAMERA_LOST
        return SESSION_COMPLETE

    def diagnostics(self):
        """Pipeline configuration and per-stage timing for this session."""
        return {
            "pose_settings": self.pose_config,
            "threaded": self.threaded,
            "frame_size": self.frame_size,
            "camera_read_failures": self.read_failures_total,
            "timing": {
                stage: _timing_summary(values)
                for stage, values in self._timings.items()
            },
        }

    def get_stats(self):
        # The analyzer must not change while it is summarised.
        self._stop_worker()
        stats = self.analyzer.summary()
        stats["experimental_posture"] = self.posture_monitor.summary()
        stats["compensation"] = self.compensation.summary()
        stats["analysis_halted"] = self.analysis_halted
        stats["last_error"] = self.last_error
        stats["session_status"] = self.session_status()
        stats["diagnostics"] = self.diagnostics()
        return stats

    def stop(self):

        self._stop_worker()
        self.timer.stop()

        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

        if self.pose is not None:
            self.pose.close()
            self.pose = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.video_label.clear()
        self.video_label.setText(
            "Camera stopped"
        )

        self.status_label.setText(
            "Camera not started"
        )

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
