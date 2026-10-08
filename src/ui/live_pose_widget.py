import cv2
import mediapipe as mp

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.utils.angles import calculate_angle
from src.action_recognition.rep_counter import RepCounter


class LivePoseWidget(QWidget):

    def __init__(self, affected_side="LEFT", parent=None):
        super().__init__(parent)

        self.affected_side = affected_side.upper()

        self.camera = None
        self.pose = None

        self.rep_counter = RepCounter()
        self.rep_counter.stage = "down"

        self.up_frames = 0
        self.down_frames = 0


        self.current_angle = 0
        self.max_angle = 0
        self.min_angle = 180

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
        self.status_label.setStyleSheet("""
            color: #374151;
            font-size: 14px;
            font-weight: 600;
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start(self):

        if self.camera is not None:
            return

        self.camera = cv2.VideoCapture(
            0,
            cv2.CAP_V4L2
        )

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

        self.rep_counter = RepCounter()
        self.rep_counter.stage = "down"

        self.up_frames = 0
        self.down_frames = 0        

        self.current_angle = 0
        self.max_angle = 0
        self.min_angle = 180

        self.status_label.setText(
            f"Monitoring {self.affected_side.lower()} arm"
        )

        self.timer.start(30)

    def update_frame(self):

        if self.camera is None:
            return

        success, frame = self.camera.read()

        if not success:
            return

        # Process original frame first
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(rgb)

        display = frame.copy()

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            mp.solutions.drawing_utils.draw_landmarks(
                display,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )

            if self.affected_side == "LEFT":
                shoulder = 11
                elbow = 13
                wrist = 15
                hip = 23

            else:
                shoulder = 12
                elbow = 14
                wrist = 16
                hip = 24

            required = [
                landmarks[shoulder],
                landmarks[elbow],
                landmarks[wrist],
                landmarks[hip],
            ]

            visible = all(
                point.visibility > 0.5
                for point in required
            )

            if visible:

                shoulder_point = (
                    landmarks[shoulder].x,
                    landmarks[shoulder].y,
                )

                elbow_point = (
                    landmarks[elbow].x,
                    landmarks[elbow].y,
                )

                hip_point = (
                    landmarks[hip].x,
                    landmarks[hip].y,
                )

                # Shoulder flexion angle:
                # hip -> shoulder -> elbow
                angle = calculate_angle(
                    hip_point,
                    shoulder_point,
                    elbow_point,
                )

                self.current_angle = angle

                self.max_angle = max(
                    self.max_angle,
                    angle,
                )

                self.min_angle = min(
                    self.min_angle,
                    angle,
                )

                # Shoulder flexion rep counting
                # Shoulder flexion rep counting
                #
                # Down position  ≈ 30–45°
                # Up position    ≈ 80–90°
                #
                # One complete cycle:
                # DOWN → UP → DOWN = 1 rep

                # Shoulder flexion rep counting
                #
                # Require several consecutive frames in a position
                # so a single MediaPipe landmark glitch does not
                # create a false repetition.

                if angle >= 80:
                    self.up_frames += 1
                    self.down_frames = 0

                    if (
                        self.up_frames >= 5
                        and self.rep_counter.stage == "down"
                    ):
                        self.rep_counter.stage = "up"

                elif angle <= 45:
                    self.down_frames += 1
                    self.up_frames = 0

                    if (
                        self.down_frames >= 5
                        and self.rep_counter.stage == "up"
                    ):
                        self.rep_counter.stage = "down"
                        self.rep_counter.reps += 1

                cv2.putText(
                    display,
                    f"Angle: {angle:.1f}",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    display,
                    f"Reps: {self.rep_counter.reps}",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

            else:

                cv2.putText(
                    display,
                    "Keep affected arm visible",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )

        else:

            cv2.putText(
                display,
                "Pose not detected",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
            )

        # Mirror for user-facing display
        display = cv2.flip(display, 1)

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

        self.status_label.setText(
            f"Reps: {self.rep_counter.reps}    "
            f"Angle: {self.current_angle:.1f}°"
        )

    def get_stats(self):

        return {
            "reps": self.rep_counter.reps,
            "max_angle": self.max_angle,
            "min_angle": self.min_angle,
        }

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
