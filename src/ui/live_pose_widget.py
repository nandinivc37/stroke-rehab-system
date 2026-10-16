import logging
import sys
import time
from collections import deque

import cv2
import mediapipe as mp

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.adaptation.policy import PolicyConfig
from src.compensation_detection.geometric_measures import (
    ExperimentalPostureMonitor,
)
from src.exercises.analyzer import (
    ExerciseAnalyzer,
    STATUS_NO_POSE,
    STATUS_NOT_VISIBLE,
    STATUS_OK,
    STATUS_UNMEASURABLE,
)
from src.exercises.legacy import LEGACY_SHOULDER_RAISE_2D

logger = logging.getLogger(__name__)

# The Kinect-trained BiLSTM compensation model is intentionally NOT used
# here: its features are incompatible with MediaPipe landmarks (see
# README, "Compensation detection status").

# Stop analysing after this many consecutive frames fail, rather than
# silently continuing with a broken pipeline.
MAX_CONSECUTIVE_ERRORS = 30
MAX_CONSECUTIVE_READ_FAILURES = 60

# Frames used for the "now" frame rate shown live (about 2-3 s at the
# rates seen on a CPU). The session average is the analyzer's mean_fps,
# the same value the adaptive policy's frame-rate gate checks.
RECENT_FPS_FRAMES = 20

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


class LivePoseWidget(QWidget):

    def __init__(
        self,
        exercise=None,
        affected_side="LEFT",
        parameters=None,
        parent=None,
    ):
        super().__init__(parent)

        self.exercise = exercise or LEGACY_SHOULDER_RAISE_2D
        self.affected_side = affected_side.upper()

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

        self.camera = None
        self.pose = None

        self.consecutive_errors = 0
        self.consecutive_read_failures = 0
        self.last_error = None
        self.analysis_halted = False

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

        layout.addWidget(self.video_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.status_label)

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

        if not self.camera.isOpened():
            self.status_label.setText(
                "Could not open camera"
            )
            self.camera = None
            return

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.analyzer.reset()
        self.posture_monitor.reset()
        self.frame_times.clear()
        self.consecutive_errors = 0
        self.consecutive_read_failures = 0
        self.last_error = None
        self.analysis_halted = False

        self.status_label.setText(
            f"Monitoring {self.affected_side.lower()} arm"
        )

        self.timer.start(30)

    # -----------------------------------------------------
    # Per-frame processing
    # -----------------------------------------------------

    def update_frame(self):

        if self.camera is None or self.analysis_halted:
            return

        success, frame = self.camera.read()

        if not success:
            self.consecutive_read_failures += 1

            if (
                self.consecutive_read_failures
                == MAX_CONSECUTIVE_READ_FAILURES
            ):
                self.status_label.setText(
                    "Camera is not delivering frames. "
                    "Check the connection and restart the session."
                )
            return

        self.consecutive_read_failures = 0
        self.frame_times.append(time.monotonic())

        # A Qt slot must never raise: PyQt6 aborts the whole application
        # on an unhandled exception. Errors are logged, counted in the
        # session summary, and shown; no substitute values are produced.
        try:
            self._process_frame(frame)
            self.consecutive_errors = 0

        except Exception as error:
            logger.exception("Frame processing failed")

            self.analyzer.record_error()
            self.consecutive_errors += 1
            self.last_error = f"{type(error).__name__}: {error}"

            if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self.analysis_halted = True
                self.timer.stop()
                self.status_label.setText(
                    "Analysis stopped after repeated processing errors "
                    f"({self.last_error}). End the session; "
                    "data recorded so far is kept."
                )
            else:
                self.status_label.setText(
                    f"Frame processing error: {self.last_error}"
                )

        self._update_fps_label()

    def recent_fps(self):
        """Processed frames per second over the last few frames."""
        if len(self.frame_times) < 2:
            return None
        span = self.frame_times[-1] - self.frame_times[0]
        if span <= 0:
            return None
        return (len(self.frame_times) - 1) / span

    def _update_fps_label(self):
        recent = self.recent_fps()
        average = self.analyzer.mean_fps

        if recent is None or average is None:
            self.fps_label.setText("Frame rate: measuring…")
            colour = FPS_OK_COLOUR
        else:
            text = (
                f"Frame rate: {recent:.1f} fps now · "
                f"{average:.1f} fps session average"
            )
            if average < self.min_fps:
                text += (
                    f" — below {self.min_fps:g} fps: this attempt "
                    "cannot be assessed"
                )
                colour = FPS_LOW_COLOUR
            else:
                colour = FPS_OK_COLOUR
            self.fps_label.setText(text)

        self.fps_label.setStyleSheet(FPS_STYLE.format(colour=colour))

    def _process_frame(self, frame):

        # MediaPipe receives the original (unmirrored) frame.
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(rgb)

        analysis = self.analyzer.update(
            results,
            time.monotonic(),
        )

        self.posture_monitor.update(
            results,
            analysis.movement_active,
        )

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
        self._show(display)

        self.status_label.setText(self._status_text())

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
        elif analysis.status == STATUS_UNMEASURABLE:
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

    def get_stats(self):
        stats = self.analyzer.summary()
        stats["experimental_posture"] = self.posture_monitor.summary()
        stats["analysis_halted"] = self.analysis_halted
        stats["last_error"] = self.last_error
        return stats

    def stop(self):

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
