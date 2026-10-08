from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.database.db import (
    get_all_patients,
    get_latest_approved_exercise,
    get_affected_areas,
    save_session,
)

from src.ui.reference_skeleton import (
    ReferenceSkeletonWidget,
)

from src.ui.live_pose_widget import (
    LivePoseWidget,
)


class RehabPage(QWidget):

    def __init__(self):
        super().__init__()

        self.current_patient_id = None
        self.current_exercise = None
        self.current_side = "LEFT"

        self.reference_widget = None
        self.live_widget = None

        self.build_ui()
        self.load_patients()

    def build_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            35, 30, 35, 30
        )

        layout.setSpacing(18)

        # -------------------------------------------------
        # Header
        # -------------------------------------------------

        title = QLabel(
            "Rehabilitation Session"
        )

        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Perform the approved exercise while "
            "following the reference movement."
        )

        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # -------------------------------------------------
        # Patient / exercise selection
        # -------------------------------------------------

        control_card = QFrame()

        control_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
        """)

        control_layout = QHBoxLayout(
            control_card
        )

        control_layout.setContentsMargins(
            18, 14, 18, 14
        )

        patient_label = QLabel(
            "Patient:"
        )

        self.patient_combo = QComboBox()

        self.patient_combo.setMinimumWidth(
            220
        )

        self.patient_combo.currentIndexChanged.connect(
            self.patient_changed
        )

        self.exercise_label = QLabel(
            "Exercise: —"
        )

        self.exercise_label.setStyleSheet("""
            font-size: 15px;
            font-weight: 700;
            color: #172554;
        """)

        self.side_label = QLabel(
            "Affected side: —"
        )

        self.start_button = QPushButton(
            "Start Exercise"
        )

        self.start_button.setObjectName(
            "primaryButton"
        )

        self.start_button.clicked.connect(
            self.start_exercise
        )

        self.stop_button = QPushButton(
            "End Session"
        )

        self.stop_button.setEnabled(False)

        self.stop_button.clicked.connect(
            self.stop_exercise
        )

        control_layout.addWidget(
            patient_label
        )

        control_layout.addWidget(
            self.patient_combo
        )

        control_layout.addSpacing(20)

        control_layout.addWidget(
            self.exercise_label
        )

        control_layout.addSpacing(15)

        control_layout.addWidget(
            self.side_label
        )

        control_layout.addStretch()

        control_layout.addWidget(
            self.start_button
        )

        control_layout.addWidget(
            self.stop_button
        )

        layout.addWidget(control_card)

        # -------------------------------------------------
        # Main exercise area
        # -------------------------------------------------

        self.exercise_area = QHBoxLayout()

        self.exercise_area.setSpacing(18)

        # Reference card
        self.reference_card = QFrame()

        self.reference_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
        """)

        reference_layout = QVBoxLayout(
            self.reference_card
        )

        reference_title = QLabel(
            "REFERENCE MOVEMENT"
        )

        reference_title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        reference_title.setStyleSheet("""
            color: #172554;
            font-size: 15px;
            font-weight: 700;
        """)

        self.reference_placeholder = QLabel(
            "Approved exercise reference\n"
            "will appear here."
        )

        self.reference_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.reference_placeholder.setStyleSheet("""
            color: #6b7280;
            font-size: 14px;
        """)

        reference_layout.addWidget(
            reference_title
        )

        reference_layout.addWidget(
            self.reference_placeholder
        )

        # Live camera card
        self.camera_card = QFrame()

        self.camera_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
        """)

        camera_layout = QVBoxLayout(
            self.camera_card
        )

        camera_title = QLabel(
            "YOUR MOVEMENT"
        )

        camera_title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        camera_title.setStyleSheet("""
            color: #172554;
            font-size: 15px;
            font-weight: 700;
        """)

        self.camera_placeholder = QLabel(
            "Camera will appear here\n"
            "when the session starts."
        )

        self.camera_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.camera_placeholder.setStyleSheet("""
            color: #6b7280;
            font-size: 14px;
        """)

        camera_layout.addWidget(
            camera_title
        )

        camera_layout.addWidget(
            self.camera_placeholder
        )

        self.exercise_area.addWidget(
            self.reference_card,
            1,
        )

        self.exercise_area.addWidget(
            self.camera_card,
            1,
        )

        layout.addLayout(
            self.exercise_area,
            1,
        )

        # -------------------------------------------------
        # Session information
        # -------------------------------------------------

        self.session_status = QLabel(
            "Select a patient with an approved "
            "exercise to begin."
        )

        self.session_status.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.session_status.setStyleSheet("""
            color: #374151;
            font-size: 14px;
            font-weight: 600;
            padding: 8px;
        """)

        layout.addWidget(
            self.session_status
        )

    # -----------------------------------------------------
    # Patient loading
    # -----------------------------------------------------

    def load_patients(self):

        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()

        patients = get_all_patients()

        for patient in patients:

            patient_id = patient[0]
            name = patient[1]

            self.patient_combo.addItem(
                f"{patient_id} - {name}",
                patient_id,
            )

        self.patient_combo.blockSignals(False)

        if self.patient_combo.count() > 0:
            self.patient_changed(0)

    # -----------------------------------------------------
    # Patient changed
    # -----------------------------------------------------

    def patient_changed(self, index):

        if index < 0:
            return

        patient_id = (
            self.patient_combo.itemData(index)
        )

        if patient_id is None:
            return

        self.current_patient_id = patient_id

        recommendation = (
            get_latest_approved_exercise(
                patient_id
            )
        )

        if recommendation is None:

            self.current_exercise = None

            self.exercise_label.setText(
                "Exercise: No approved exercise"
            )

            self.side_label.setText(
                "Affected side: —"
            )

            self.session_status.setText(
                "No approved exercise found. "
                "Go to Assessment and approve an exercise first."
            )

            return

        self.current_exercise = (
            recommendation[1]
        )

        areas = get_affected_areas(
            patient_id
        )

        if areas:

            self.current_side = (
                areas[0][1].upper()
            )

            if self.current_side == "BILATERAL":
                self.current_side = "LEFT"

        else:
            self.current_side = "LEFT"

        self.exercise_label.setText(
            f"Exercise: {self.current_exercise}"
        )

        self.side_label.setText(
            f"Affected side: {self.current_side}"
        )

        self.session_status.setText(
            "Exercise approved. "
            "Press Start Exercise when ready."
        )

    # -----------------------------------------------------
    # Start
    # -----------------------------------------------------

    def start_exercise(self):

        if self.current_patient_id is None:
            QMessageBox.warning(
                self,
                "No Patient",
                "Please select a patient.",
            )
            return

        if not self.current_exercise:
            QMessageBox.warning(
                self,
                "No Approved Exercise",
                "This patient does not have an approved exercise.",
            )
            return

        # ---------------------------------------------
        # Reference skeleton
        # ---------------------------------------------

        self.reference_widget = (
            ReferenceSkeletonWidget(
                exercise_name=self.current_exercise,
                affected_side=self.current_side,
            )
        )

        self.reference_card.layout().replaceWidget(
            self.reference_placeholder,
            self.reference_widget,
        )

        self.reference_placeholder.deleteLater()
        self.reference_placeholder = None

        # ---------------------------------------------
        # Live camera
        # ---------------------------------------------

        self.live_widget = LivePoseWidget(
            affected_side=self.current_side
        )

        self.camera_card.layout().replaceWidget(
            self.camera_placeholder,
            self.live_widget,
        )

        self.camera_placeholder.deleteLater()
        self.camera_placeholder = None

        self.live_widget.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.session_status.setText(
            f"Performing {self.current_exercise}. "
            f"Follow the reference movement."
        )

    # -----------------------------------------------------
    # Stop
    # -----------------------------------------------------

    def stop_exercise(self):

        if self.live_widget is None:
            return

        stats = self.live_widget.get_stats()

        self.live_widget.stop()

        save_session(
            self.current_patient_id,
            self.current_exercise,
            stats["reps"],
            stats["max_angle"],
            stats["min_angle"],
        )

        # Remove live widget
        self.camera_card.layout().removeWidget(
            self.live_widget
        )

        self.live_widget.deleteLater()
        self.live_widget = None

        self.camera_placeholder = QLabel(
            "Camera stopped.\n"
            "Start another session when ready."
        )

        self.camera_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.camera_placeholder.setStyleSheet("""
            color: #6b7280;
            font-size: 14px;
        """)

        self.camera_card.layout().addWidget(
            self.camera_placeholder
        )

        # Remove reference widget
        if self.reference_widget is not None:

            self.reference_widget.pause()

            self.reference_card.layout().removeWidget(
                self.reference_widget
            )

            self.reference_widget.deleteLater()
            self.reference_widget = None

        self.reference_placeholder = QLabel(
            "Approved exercise reference\n"
            "will appear here."
        )

        self.reference_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.reference_placeholder.setStyleSheet("""
            color: #6b7280;
            font-size: 14px;
        """)

        self.reference_card.layout().addWidget(
            self.reference_placeholder
        )

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.session_status.setText(
            f"Session saved — "
            f"{stats['reps']} repetitions."
        )
