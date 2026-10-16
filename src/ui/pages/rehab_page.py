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

import logging

from src.adaptation.workflow import propose_after_session
from src.database.db import (
    get_active_prescription,
    get_all_patients,
    get_affected_areas,
    save_session_summary,
)
from src.exercises.catalogue import get_exercise_config

logger = logging.getLogger(__name__)

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
        self.current_config = None
        self.current_parameters = None
        self.current_prescription_id = None
        self.current_side = "LEFT"
        self.last_session_id = None
        self.last_adaptation = None

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
            QLabel { border: none; }
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

        # Prescribed parameters and instructions for the exercise.
        self.prescription_label = QLabel("")
        self.prescription_label.setWordWrap(True)
        self.prescription_label.setStyleSheet("""
            color: #374151;
            font-size: 13px;
            padding: 2px 4px;
        """)

        layout.addWidget(self.prescription_label)

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
            QLabel { border: none; }
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
            QLabel { border: none; }
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

        self.session_status.setWordWrap(True)

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
        else:
            # No patients (e.g. the last one was deleted).
            self.current_patient_id = None
            self._clear_prescription(
                "Exercise: —", "Register a patient to begin."
            )

    # -----------------------------------------------------
    # Patient changed
    # -----------------------------------------------------

    def refresh(self):
        """Reload patients when the page is shown (not mid-session)."""
        if self.live_widget is None:
            current = self.current_patient_id
            self.load_patients()
            if current is not None:
                index = self.patient_combo.findData(current)
                if index >= 0:
                    self.patient_combo.setCurrentIndex(index)

    def _clear_prescription(self, exercise_text, status_text):
        self.current_exercise = None
        self.current_config = None
        self.current_parameters = None
        self.current_prescription_id = None

        self.exercise_label.setText(exercise_text)
        self.side_label.setText("Affected side: —")
        self.prescription_label.setText("")
        self.session_status.setText(status_text)
        self.start_button.setEnabled(False)

    def patient_changed(self, index):

        if index < 0:
            return

        patient_id = (
            self.patient_combo.itemData(index)
        )

        if patient_id is None:
            return

        self.current_patient_id = patient_id

        prescription = get_active_prescription(patient_id)

        if prescription is None:
            self._clear_prescription(
                "Exercise: No approved exercise",
                "No approved exercise found. "
                "Go to Assessment and approve an exercise first.",
            )
            return

        config = get_exercise_config(
            prescription.get("exercise_id") or prescription["exercise"]
        )

        if config is None or not config.implemented:
            self._clear_prescription(
                f"Exercise: {prescription['exercise']}",
                f"'{prescription['exercise']}' is approved but is not "
                "supported by the movement-analysis system. Approve a "
                "supported exercise on the Assessment page.",
            )
            return

        side = prescription.get("affected_side")

        if not side:
            areas = get_affected_areas(patient_id)
            side = areas[0][1] if areas else "LEFT"

        side = side.upper()

        # Bilateral impairment: the analysis follows one arm at a time;
        # the original application defaulted to the left arm.
        if side not in config.sides:
            side = "LEFT"

        self.current_side = side
        self.current_config = config
        self.current_exercise = config.display_name
        self.current_prescription_id = prescription["recommendation_id"]
        self.current_parameters = config.resolve_parameters(
            prescription["parameters"]
        )

        self.exercise_label.setText(
            f"Exercise: {config.display_name}"
        )

        self.side_label.setText(
            f"Affected side: {self.current_side}"
        )

        targets = ", ".join(
            f"{p.label}: {self.current_parameters[p.name]:g} {p.unit}"
            for p in config.adaptive_parameters
        )
        source = "" if prescription["parameters"] else " (default values)"

        self.prescription_label.setText(
            f"{targets}{source}. {config.instructions} "
            f"Camera: {config.camera_view}."
        )

        self.start_button.setEnabled(True)

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

        if self.current_config is None:
            QMessageBox.warning(
                self,
                "No Approved Exercise",
                "This patient does not have an approved, supported "
                "exercise.",
            )
            return

        # ---------------------------------------------
        # Live camera (created first: it owns the analyzer whose
        # reference movement is displayed and used for scoring)
        # ---------------------------------------------

        self.live_widget = LivePoseWidget(
            exercise=self.current_config,
            affected_side=self.current_side,
            parameters=self.current_parameters,
        )

        self.camera_card.layout().replaceWidget(
            self.camera_placeholder,
            self.live_widget,
        )

        self.camera_placeholder.deleteLater()
        self.camera_placeholder = None

        # ---------------------------------------------
        # Reference skeleton
        # ---------------------------------------------

        reference = self.live_widget.analyzer.reference

        self.reference_widget = (
            ReferenceSkeletonWidget(
                exercise_name=self.current_exercise,
                affected_side=self.current_side,
                frames=reference.frames,
                caption=(
                    f"Illustrative synthetic reference "
                    f"({self.current_config.reference_view} view)"
                    if reference.synthetic else None
                ),
            )
        )

        self.reference_card.layout().replaceWidget(
            self.reference_placeholder,
            self.reference_widget,
        )

        self.reference_placeholder.deleteLater()
        self.reference_placeholder = None

        self.live_widget.start()

        if self.live_widget.camera is None:
            # Camera could not be opened: nothing will be recorded.
            self._teardown_session_widgets(
                "Camera could not be opened.\n"
                "Connect a webcam and try again."
            )
            self.session_status.setText(
                "Camera could not be opened. No session was started."
            )
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.patient_combo.setEnabled(False)

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

        self._teardown_session_widgets(
            "Camera stopped.\n"
            "Start another session when ready."
        )

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.patient_combo.setEnabled(True)

        if stats["frames_total"] == 0:
            self.session_status.setText(
                "No camera frames were processed, so the session was "
                "not saved."
            )
            return

        self.last_session_id = save_session_summary(
            self.current_patient_id,
            self.current_exercise,
            stats,
            prescription_id=self.current_prescription_id,
        )

        self.last_adaptation = None
        adaptation_text = ""

        try:
            self.last_adaptation = propose_after_session(
                self.current_patient_id,
                self.last_session_id,
            )
        except Exception as error:
            # The session itself is already saved; report the failure
            # rather than inventing a decision.
            logger.exception("Adaptation evaluation failed")
            adaptation_text = (
                f" | Adaptation evaluation failed: {error}"
            )

        if self.last_adaptation is not None:
            adaptation_text = "\n" + self.last_adaptation["feedback"]
            if self.last_adaptation["proposal_id"] is not None:
                adaptation_text += (
                    " (Proposed change awaiting therapist approval on the "
                    "Progress page.)"
                )

        # The next attempt runs with the ACTIVE prescription: the
        # controller's next_attempt_parameters (a proposed change only
        # applies once a therapist approves it).
        self._reload_prescription()

        self.session_status.setText(
            self._summary_text(stats) + adaptation_text
        )

    def _reload_prescription(self):
        index = self.patient_combo.findData(self.current_patient_id)
        if index >= 0:
            self.patient_changed(index)

    def _summary_text(self, stats):
        target = stats.get("target_reps")
        reps = (
            f"{stats['reps']} / {target} repetitions"
            if target else f"{stats['reps']} repetitions"
        )

        similarity = stats["trajectory_similarity"]
        similarity_text = (
            f"{similarity:.1f}%"
            if similarity is not None
            else "not available (no complete movement)"
        )

        return (
            f"Session saved — {reps} | "
            f"Movement similarity: {similarity_text}"
            f"{self._tracking_text(stats.get('tracking'))}"
        )

    @staticmethod
    def _tracking_text(tracking):
        """Tracking breakdown, so an unassessable session is explainable."""
        if not tracking or not tracking["frames_total"]:
            return ""
        logger.info("Session tracking: %s", tracking)
        total = tracking["frames_total"]
        text = (
            f" | Arm tracked in {100 * tracking['measured_fraction']:.0f}% "
            f"of {total} frames (no person detected: "
            f"{100 * tracking['frames_no_pose'] / total:.0f}%, required "
            f"landmarks not visible: "
            f"{100 * tracking['frames_not_visible'] / total:.0f}%)"
        )
        missing = tracking["not_visible_by_landmark"]
        if missing:
            name, count = next(iter(missing.items()))
            text += (
                f"; most often not visible: {name} "
                f"({100 * count / total:.0f}% of frames)"
            )
        return text

    def _teardown_session_widgets(self, camera_message):

        if self.live_widget is not None:
            self.camera_card.layout().removeWidget(self.live_widget)
            self.live_widget.deleteLater()
            self.live_widget = None

        if self.camera_placeholder is None:
            self.camera_placeholder = QLabel(camera_message)
            self.camera_placeholder.setAlignment(
                Qt.AlignmentFlag.AlignCenter
            )
            self.camera_placeholder.setStyleSheet("""
                color: #6b7280;
                font-size: 14px;
            """)
            self.camera_card.layout().addWidget(self.camera_placeholder)

        if self.reference_widget is not None:
            self.reference_widget.pause()
            self.reference_card.layout().removeWidget(
                self.reference_widget
            )
            self.reference_widget.deleteLater()
            self.reference_widget = None

        if self.reference_placeholder is None:
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
