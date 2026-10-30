import logging
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
    ExerciseAnalyzer,
    STATUS_IMPLAUSIBLE,
    STATUS_NO_POSE,
    STATUS_NOT_VISIBLE,
    STATUS_OK,
    STATUS_UNMEASURABLE,
)
from src.exercises.legacy import LEGACY_SHOULDER_RAISE_2D
from src.pose_estimation.config import pose_settings

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

        layout.addWidget(self.video_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.compensation_label)
        layout.addWidget(self.status_label)

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
                return {
                    "display": None,
                    "status": (
                        "Camera is not delivering frames. "
                        "Check the connection and restart the session."
                    ),
                    "fps": None,
                }
            return None

        timestamp = self._timestamp()
        self.consecutive_read_failures = 0
        self.frame_times.append(timestamp)
        self._timings["read"].append(read_time)
        self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))

        # A Qt slot must never raise: PyQt6 aborts the whole application
        # on an unhandled exception. Errors are logged, counted in the
        # session summary, and shown; no substitute values are produced.
        try:
            display, status = self._process_frame(frame, timestamp)
            self.consecutive_errors = 0

        except Exception as error:
            logger.exception("Frame processing failed")

            self.analyzer.record_error(timestamp)
            self.consecutive_errors += 1
            self.last_error = f"{type(error).__name__}: {error}"
            display = None

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
                "compensation": self._compensation_feedback}

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

        analysis = self.analyzer.update(
            results,
            timestamp,
        )

        self.posture_monitor.update(
            results,
            analysis.movement_active,
        )

        self._update_compensation(results, analysis.movement_active)

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

    def _update_compensation(self, results, movement_active):
        """
        Compensation check for this frame. Its failure must not stop
        repetition counting or DTW, so errors only disable the check.
        """
        try:
            frame = self.compensation.update(results, movement_active)
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
        self._display_pending = False

        if output.get("status"):
            self.status_label.setText(output["status"])

        if output.get("fps") is not None:
            text, colour = output["fps"]
            self.fps_label.setText(text)
            self.fps_label.setStyleSheet(FPS_STYLE.format(colour=colour))

        if output.get("compensation") is not None:
            self._set_compensation_label(*output["compensation"])

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
        )

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
