# src/ui/reference_skeleton.py

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QPen, QBrush
from PyQt6.QtWidgets import QWidget

from src.assessment.exercise_library import (
    get_reference_sequence,
)


# =========================================================
# MediaPipe Pose connections
# =========================================================

POSE_CONNECTIONS = [
    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Shoulders
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28),

    # Feet
    (27, 31),
    (28, 32),
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
        self.affected_side = affected_side

        # ---------------------------------------------
        # Load reference sequence
        # ---------------------------------------------

        self.sequence = get_reference_sequence(
            exercise_name,
            affected_side,
        )

        self.frame_index = 0

        # ---------------------------------------------
        # Timer for animation
        # ---------------------------------------------

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.next_frame
        )

        # ~12 FPS
        self.timer.start(80)

        # ---------------------------------------------
        # Widget appearance
        # ---------------------------------------------

        self.setMinimumSize(350, 400)

        self.setStyleSheet("""
            background-color: #f8fafc;
            border: 1px solid #dbeafe;
            border-radius: 10px;
        """)

    # =====================================================
    # Animation
    # =====================================================

    def next_frame(self):

        self.frame_index += 1

        if self.frame_index >= len(self.sequence):
            self.frame_index = 0

        self.update()

    # =====================================================
    # Convert normalized coordinates to widget coordinates
    # =====================================================

    def landmark_to_pixel(
        self,
        x,
        y,
    ):

        width = self.width()
        height = self.height()

        # Leave some space around skeleton
        margin_x = 45
        margin_y = 30

        pixel_x = (
            margin_x
            + x * (width - 2 * margin_x)
        )

        pixel_y = (
            margin_y
            + y * (height - 2 * margin_y)
        )

        return pixel_x, pixel_y

    # =====================================================
    # Paint skeleton
    # =====================================================

    def paintEvent(self, event):

        if self.sequence is None:
            return

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        # ---------------------------------------------
        # Background
        # ---------------------------------------------

        painter.fillRect(
            self.rect(),
            QBrush(Qt.GlobalColor.white),
        )

        # ---------------------------------------------
        # Current frame
        # ---------------------------------------------

        frame = self.sequence[
            self.frame_index
        ]

        # ---------------------------------------------
        # Draw connections
        # ---------------------------------------------

        bone_pen = QPen(
            Qt.GlobalColor.darkBlue
        )

        bone_pen.setWidth(5)

        painter.setPen(bone_pen)

        for start, end in POSE_CONNECTIONS:

            x1, y1 = frame[start]
            x2, y2 = frame[end]

            px1, py1 = self.landmark_to_pixel(
                float(x1),
                float(y1),
            )

            px2, py2 = self.landmark_to_pixel(
                float(x2),
                float(y2),
            )

            painter.drawLine(
                int(px1),
                int(py1),
                int(px2),
                int(py2),
            )

        # ---------------------------------------------
        # Draw landmarks
        # ---------------------------------------------

        painter.setPen(Qt.PenStyle.NoPen)

        painter.setBrush(
            QBrush(Qt.GlobalColor.blue)
        )

        radius = 6

        for x, y in frame:

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

    # =====================================================
    # Public controls
    # =====================================================

    def pause(self):

        self.timer.stop()

    def resume(self):

        if not self.timer.isActive():
            self.timer.start(80)

    def reset(self):

        self.frame_index = 0
        self.update()