import cv2
import math
import mediapipe as mp
import numpy as np

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

# from src.utils.angles import calculate_angle
from src.action_recognition.rep_counter import RepCounter

from src.quality_assessment.dtw import dtw_similarity
from src.assessment.exercise_library import get_reference_sequence


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

        self.trajectory = []


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

        self.trajectory = []  # Reset trajectory for every new session
      

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
            #

            if self.affected_side == "LEFT":
                shoulder = 11
                elbow = 13
                wrist = 15
            else:
                shoulder = 12
                elbow = 14
                wrist = 16

            required = [
                landmarks[shoulder],
                landmarks[elbow],
                landmarks[wrist],
            ]

            visible = all(
                point.visibility > 0.5
                for point in required
            )

            if visible:

                shoulder_x = landmarks[shoulder].x
                shoulder_y = landmarks[shoulder].y

                wrist_x = landmarks[wrist].x
                wrist_y = landmarks[wrist].y

                elbow_x = landmarks[elbow].x
                elbow_y = landmarks[elbow].y

                # -----------------------------------------
                # Normalize arm trajectory
                # -----------------------------------------

                elbow_rel_x = elbow_x - shoulder_x
                elbow_rel_y = elbow_y - shoulder_y

                wrist_rel_x = wrist_x - shoulder_x
                wrist_rel_y = wrist_y - shoulder_y

                # Upper-arm length used for scale normalization
                arm_length = (
                    elbow_rel_x ** 2
                    + elbow_rel_y ** 2
                ) ** 0.5

                # Vector from shoulder to wrist
                dx = wrist_x - shoulder_x
                dy = wrist_y - shoulder_y

                # Arm elevation relative to the
                # vertical downward direction.
                arm_length = (dx ** 2 + dy ** 2) ** 0.5

                if arm_length > 0:
                    cosine = dy / arm_length
                    cosine = max(-1.0, min(1.0, cosine))

                    angle = math.degrees(
                        math.acos(cosine)
                    )
                else:
                    angle = 0

                self.trajectory.append(float(angle))

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
        reference_sequence = get_reference_sequence(
            "Shoulder Flexion",
            self.affected_side,
        )

        if self.affected_side == "LEFT":
            shoulder = 11
            wrist = 15
        else:
            shoulder = 12
            wrist = 16

        reference_trajectory = []

        for frame in reference_sequence:
            shoulder_x, shoulder_y = frame[shoulder]
            wrist_x, wrist_y = frame[wrist]

            dx = wrist_x - shoulder_x
            dy = wrist_y - shoulder_y

            arm_length = (dx ** 2 + dy ** 2) ** 0.5

            if arm_length > 0:
                cosine = dy / arm_length
                cosine = max(-1.0, min(1.0, cosine))
                angle = math.degrees(math.acos(cosine))
                reference_trajectory.append(angle)

        trajectory_similarity = 0.0
        range_completion = 0.0
        movement_range = self.max_angle - self.min_angle

        print(
            f"DEBUG | min={self.min_angle:.1f}, "
            f"max={self.max_angle:.1f}, "
            f"range={movement_range:.1f}, "
            f"reps={self.rep_counter.reps}, "
            f"frames={len(self.trajectory)}"
        )

        reference_peak = max(reference_trajectory, default=0.0)
        patient_peak = max(self.trajectory, default=0.0)

        # Reject sessions with virtually no arm movement.
        if len(self.trajectory) > 5 and movement_range >= 12:
            dtw_score = dtw_similarity(
                reference_trajectory,
                self.trajectory,
            )

            if reference_peak > 0:
                range_completion = min(
                    1.0,
                    patient_peak / reference_peak,
                )

            # Combine trajectory matching and range completion.
            trajectory_similarity = (
                0.5 * dtw_score
                + 0.5 * range_completion * 100.0
            )

            trajectory_similarity = round(
                max(0.0, min(100.0, trajectory_similarity)),
                2,
            )

        print(
            f"Peak angle: {patient_peak:.1f}° | "
            f"Reference peak: {reference_peak:.1f}° | "
            f"Range completion: {range_completion * 100:.1f}% | "
            f"Similarity: {trajectory_similarity:.1f}%"
        )

        return {
            "reps": self.rep_counter.reps,
            "max_angle": self.max_angle,
            "min_angle": self.min_angle,
            "trajectory": self.trajectory,
            "trajectory_similarity": trajectory_similarity,
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
