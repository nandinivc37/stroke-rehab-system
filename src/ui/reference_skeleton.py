from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QPen, QBrush
from PyQt6.QtWidgets import QWidget

from src.assessment.exercise_library import get_reference_sequence


# Only draw landmarks that we actually define in our reference pose.
POSE_CONNECTIONS = [
    # Head / neck
    ("nose", "neck"),

    # Shoulders
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),

    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    # Upper torso
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


class ReferenceSkeletonWidget(QWidget):

    def __init__(
        self,
        exercise_name="Shoulder Flexion",
        affected_side="LEFT",
        parent=None,
    ):
        super().__init__(parent)

        self.exercise_name = exercise_name
        self.affected_side = affected_side.upper()

        self.sequence = get_reference_sequence(
            exercise_name,
            self.affected_side,
        )

        self.frame_index = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(80)

        self.setMinimumSize(350, 400)

        self.setStyleSheet("""
            background-color: #f8fafc;
            border: 1px solid #dbeafe;
            border-radius: 10px;
        """)

    def next_frame(self):
        self.frame_index += 1

        if self.frame_index >= len(self.sequence):
            self.frame_index = 0

        self.update()

    def landmark_to_pixel(self, x, y):
        width = self.width()
        height = self.height()

        margin_x = 70
        margin_y = 35

        pixel_x = margin_x + x * (width - 2 * margin_x)
        pixel_y = margin_y + y * (height - 2 * margin_y)

        return pixel_x, pixel_y

    def get_points(self, frame):

        # MediaPipe landmark indices
        nose = 0

        left_shoulder = 11
        right_shoulder = 12

        left_elbow = 13
        right_elbow = 14

        left_wrist = 15
        right_wrist = 16

        left_hip = 23
        right_hip = 24

        # left_knee = 25
        # right_knee = 26

        # left_ankle = 27
        # right_ankle = 28

        points = {
            "nose": frame[nose],

            "left_shoulder": frame[left_shoulder],
            "right_shoulder": frame[right_shoulder],

            "left_elbow": frame[left_elbow],
            "right_elbow": frame[right_elbow],

            "left_wrist": frame[left_wrist],
            "right_wrist": frame[right_wrist],

            "left_hip": frame[left_hip],
            "right_hip": frame[right_hip],

            # "left_knee": frame[left_knee],
            # "right_knee": frame[right_knee],

            # "left_ankle": frame[left_ankle],
            # "right_ankle": frame[right_ankle],
        }

        # Neck is halfway between the shoulders.
        left = points["left_shoulder"]
        right = points["right_shoulder"]

        points["neck"] = (
            (left[0] + right[0]) / 2,
            (left[1] + right[1]) / 2,
        )

        return points

    def paintEvent(self, event):

        if self.sequence is None:
            return

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        # Background
        painter.fillRect(
            self.rect(),
            QBrush(Qt.GlobalColor.white),
        )

        frame = self.sequence[self.frame_index]
        points = self.get_points(frame)

        # -------------------------------------------------
        # Draw skeleton bones
        # -------------------------------------------------

        bone_pen = QPen(Qt.GlobalColor.darkBlue)
        bone_pen.setWidth(5)
        bone_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.setPen(bone_pen)

        for start_name, end_name in POSE_CONNECTIONS:

            start = points[start_name]
            end = points[end_name]

            x1, y1 = self.landmark_to_pixel(
                float(start[0]),
                float(start[1]),
            )

            x2, y2 = self.landmark_to_pixel(
                float(end[0]),
                float(end[1]),
            )

            painter.drawLine(
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            )

        # -------------------------------------------------
        # Draw head
        # -------------------------------------------------

        nose = points["nose"]

        hx, hy = self.landmark_to_pixel(
            float(nose[0]),
            float(nose[1]),
        )

        head_radius = 18

        painter.setPen(
            QPen(Qt.GlobalColor.darkBlue, 4)
        )

        painter.setBrush(
            QBrush(Qt.GlobalColor.white)
        )

        painter.drawEllipse(
            int(hx - head_radius),
            int(hy - head_radius),
            head_radius * 2,
            head_radius * 2,
        )

        # -------------------------------------------------
        # Draw joints
        # -------------------------------------------------

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(
            QBrush(Qt.GlobalColor.blue)
        )

        radius = 6

        joint_names = [
            "nose",
            "neck",

            "left_shoulder",
            "right_shoulder",

            "left_elbow",
            "right_elbow",

            "left_wrist",
            "right_wrist",

            "left_hip",
            "right_hip",

            # "left_knee",
            # "right_knee",

            # "left_ankle",
            # "right_ankle",
        ]

        for name in joint_names:

            x, y = points[name]

            px, py = self.landmark_to_pixel(
                float(x),
                float(y),
            )

            painter.drawEllipse(
                int(px - radius),
                int(py - radius),
                radius * 2,
                radius * 2,
            )

        painter.end()

    def pause(self):
        self.timer.stop()

    def resume(self):

        if not self.timer.isActive():
            self.timer.start(80)

    def reset(self):
        self.frame_index = 0
        self.update()