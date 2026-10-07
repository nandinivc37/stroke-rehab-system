from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QTextEdit,
    QPushButton,
    QMessageBox,
    QWidget,
    QFrame,
)

from src.database.db import (
    get_all_patients,
    get_affected_areas,
    save_clinical_assessment,
    save_recommendation,
    update_recommendation_status,
)

from src.assessment.recommendation import recommend_exercise


class AssessmentPage(QWidget):

    def __init__(self):
        super().__init__()

        self.current_recommendation_id = None

        # =================================================
        # Page styling
        # =================================================

        self.setStyleSheet("""
            QLabel {
                color: #374151;
            }

            QComboBox,
            QDoubleSpinBox,
            QTextEdit {
                color: #1f2937;
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 7px;
            }

            QComboBox:focus,
            QDoubleSpinBox:focus,
            QTextEdit:focus {
                border: 1px solid #2563eb;
            }

            QComboBox QAbstractItemView {
                color: #1f2937;
                background-color: #ffffff;
                selection-color: #172554;
                selection-background-color: #dbeafe;
                border: 1px solid #d1d5db;
            }

            QPushButton {
                color: #374151;
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #f3f4f6;
                border: 1px solid #9ca3af;
            }

            QPushButton#primaryButton {
                color: #ffffff;
                background-color: #2563eb;
                border: 1px solid #2563eb;
            }

            QPushButton#primaryButton:hover {
                background-color: #1d4ed8;
            }

            QFrame#recommendationCard {
                background-color: #ffffff;
                border: 1px solid #bfdbfe;
                border-radius: 10px;
            }
        """)

        # =================================================
        # Main layout
        # =================================================

        layout = QVBoxLayout(self)
        layout.setContentsMargins(45, 35, 45, 30)
        layout.setSpacing(12)

        # =================================================
        # Header
        # =================================================

        title = QLabel("Clinical Assessment")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Enter patient-specific clinical information for exercise recommendation."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # =================================================
        # Assessment form
        # =================================================

        form = QFormLayout()
        form.setSpacing(10)

        # -------------------------------------------------
        # Patient
        # -------------------------------------------------

        self.patient_input = QComboBox()
        self.patient_input.addItem("Select patient", None)

        self.load_patients()

        self.patient_input.currentIndexChanged.connect(
            self.patient_changed
        )

        form.addRow("Patient:", self.patient_input)

        # -------------------------------------------------
        # Diagnosis
        # -------------------------------------------------

        self.diagnosis_input = QComboBox()
        self.diagnosis_input.addItem("Select diagnosis")
        self.diagnosis_input.addItems([
            "Stroke",
            "Post-stroke hemiparesis",
            "Other",
        ])

        form.addRow("Diagnosis:", self.diagnosis_input)

        # -------------------------------------------------
        # Impairment
        # -------------------------------------------------

        self.impairment_input = QComboBox()
        self.impairment_input.addItem("Select impairment")
        self.impairment_input.addItems([
            "Reduced ROM",
            "Reduced strength",
            "Poor movement coordination",
            "Movement compensation",
            "Other",
        ])

        form.addRow("Impairment:", self.impairment_input)

        # -------------------------------------------------
        # Baseline ROM
        # -------------------------------------------------

        self.rom_input = QDoubleSpinBox()
        self.rom_input.setRange(0.0, 360.0)
        self.rom_input.setDecimals(1)
        self.rom_input.setValue(0.0)
        self.rom_input.setSuffix("°")

        form.addRow("Baseline ROM:", self.rom_input)

        # -------------------------------------------------
        # Restrictions
        # -------------------------------------------------

        self.restrictions_input = QComboBox()
        self.restrictions_input.addItem("Select restriction")
        self.restrictions_input.addItems([
            "No major restriction",
            "Limited range",
            "Pain-limited movement",
            "Therapist specified",
        ])

        form.addRow("Restrictions:", self.restrictions_input)

        # -------------------------------------------------
        # Clinical notes
        # -------------------------------------------------

        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText(
            "Additional clinical observations..."
        )
        self.notes_input.setFixedHeight(70)

        form.addRow("Clinical Notes:", self.notes_input)

        layout.addLayout(form)

        # =================================================
        # Affected area
        # =================================================

        self.area_label = QLabel(
            "Affected area: Select a patient"
        )

        self.area_label.setTextFormat(
            Qt.TextFormat.RichText
        )

        self.area_label.setStyleSheet(
            "color: #374151; "
            "font-size: 14px; "
            "padding-top: 2px; "
            "padding-bottom: 2px;"
        )

        self.area_label.setWordWrap(False)

        layout.addWidget(self.area_label)
        layout.addSpacing(4)

        # =================================================
        # Save / Reset buttons
        # =================================================

        buttons = QHBoxLayout()
        buttons.addStretch()

        reset_button = QPushButton("Reset Form")
        reset_button.clicked.connect(
            self.clear_form
        )

        save_button = QPushButton("Save Assessment")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(
            self.save_assessment
        )

        buttons.addWidget(reset_button)
        buttons.addWidget(save_button)

        layout.addLayout(buttons)

        # =================================================
        # Recommendation card
        # =================================================

        self.recommendation_card = QFrame()
        self.recommendation_card.setObjectName(
            "recommendationCard"
        )

        recommendation_layout = QVBoxLayout(
            self.recommendation_card
        )

        recommendation_layout.setContentsMargins(
            20, 16, 20, 16
        )

        recommendation_layout.setSpacing(7)

        # -------------------------------------------------
        # Recommendation title
        # -------------------------------------------------

        recommendation_title = QLabel(
            "AI Exercise Recommendation"
        )

        recommendation_title.setStyleSheet(
            "font-size: 18px; "
            "font-weight: bold; "
            "color: #172554;"
        )

        recommendation_layout.addWidget(
            recommendation_title
        )

        # -------------------------------------------------
        # Exercise
        # -------------------------------------------------

        self.exercise_label = QLabel(
            "No recommendation generated yet."
        )

        self.exercise_label.setStyleSheet(
            "font-size: 17px; "
            "font-weight: bold; "
            "color: #2563eb;"
        )

        recommendation_layout.addWidget(
            self.exercise_label
        )

        # -------------------------------------------------
        # Difficulty
        # -------------------------------------------------

        self.difficulty_label = QLabel(
            "Difficulty: —"
        )

        recommendation_layout.addWidget(
            self.difficulty_label
        )

        # -------------------------------------------------
        # Reason
        # -------------------------------------------------

        self.reason_label = QLabel(
            "Complete and save a clinical assessment "
            "to generate a recommendation."
        )

        self.reason_label.setWordWrap(True)

        recommendation_layout.addWidget(
            self.reason_label
        )

        # -------------------------------------------------
        # Approval buttons
        # -------------------------------------------------

        recommendation_buttons = QHBoxLayout()
        recommendation_buttons.addStretch()

        self.reject_button = QPushButton("Reject")
        self.reject_button.clicked.connect(
            self.reject_recommendation
        )

        self.approve_button = QPushButton(
            "Approve Exercise"
        )
        self.approve_button.setObjectName(
            "primaryButton"
        )
        self.approve_button.clicked.connect(
            self.approve_recommendation
        )

        self.reject_button.setEnabled(False)
        self.approve_button.setEnabled(False)

        recommendation_buttons.addWidget(
            self.reject_button
        )

        recommendation_buttons.addWidget(
            self.approve_button
        )

        recommendation_layout.addLayout(
            recommendation_buttons
        )

        layout.addWidget(
            self.recommendation_card
        )

        layout.addStretch()

    # =====================================================
    # Load patients
    # =====================================================

    def load_patients(self):

        patients = get_all_patients()

        for patient in patients:

            patient_id = patient[0]
            name = patient[1]

            self.patient_input.addItem(
                f"{patient_id} - {name}",
                patient_id
            )

    # =====================================================
    # Patient changed
    # =====================================================

    def patient_changed(self):

        patient_id = self.patient_input.currentData()

        if patient_id is None:

            self.area_label.setText(
                '<span style="color:#374151;">'
                'Affected area:'
                '</span> '
                '<span style="color:#6b7280;">'
                'Select a patient'
                '</span>'
            )

            return

        areas = get_affected_areas(patient_id)

        if not areas:

            self.area_label.setText(
                '<span style="color:#374151;">'
                'Affected area:'
                '</span> '
                '<span style="color:#6b7280;">'
                'Not specified'
                '</span>'
            )

            return

        area = areas[0]

        side = area[1]
        body_region = area[2]
        joint = area[3]

        self.area_label.setText(
            '<span style="color:#374151;">'
            'Affected area:'
            '</span> '
            '<span style="color:#2563eb; font-weight:600;">'
            f'{side} • {body_region} • {joint}'
            '</span>'
        )

    # =====================================================
    # Save assessment
    # =====================================================

    def save_assessment(self):

        patient_id = self.patient_input.currentData()

        if patient_id is None:

            QMessageBox.warning(
                self,
                "Missing Information",
                "Please select a patient."
            )

            return

        if self.diagnosis_input.currentIndex() == 0:

            QMessageBox.warning(
                self,
                "Missing Information",
                "Please select a diagnosis."
            )

            return

        if self.impairment_input.currentIndex() == 0:

            QMessageBox.warning(
                self,
                "Missing Information",
                "Please select an impairment."
            )

            return

        if self.restrictions_input.currentIndex() == 0:

            QMessageBox.warning(
                self,
                "Missing Information",
                "Please select the movement restriction."
            )

            return

        if self.rom_input.value() <= 0:

            QMessageBox.warning(
                self,
                "Missing Information",
                "Please enter the baseline ROM."
            )

            return

        areas = get_affected_areas(patient_id)

        if not areas:

            QMessageBox.warning(
                self,
                "Missing Information",
                "This patient has no affected area recorded."
            )

            return

        area = areas[0]

        affected_side = area[1]
        body_region = area[2]
        joint = area[3]

        diagnosis = self.diagnosis_input.currentText()
        impairment = self.impairment_input.currentText()
        baseline_rom = self.rom_input.value()
        restrictions = self.restrictions_input.currentText()

        clinical_notes = (
            self.notes_input
            .toPlainText()
            .strip()
        )

        try:

            # ---------------------------------------------
            # Save clinical assessment
            # ---------------------------------------------

            assessment_id = save_clinical_assessment(
                patient_id,
                diagnosis,
                impairment,
                baseline_rom,
                restrictions,
                clinical_notes,
            )

            # ---------------------------------------------
            # Generate recommendation
            # ---------------------------------------------

            recommendation = recommend_exercise(
                diagnosis=diagnosis,
                affected_side=affected_side,
                body_region=body_region,
                joint=joint,
                impairment=impairment,
                baseline_rom=baseline_rom,
                restrictions=restrictions,
            )

            if not recommendation["success"]:

                QMessageBox.information(
                    self,
                    "Assessment Saved",
                    "Clinical assessment was saved, but no "
                    "suitable exercise is currently available "
                    "for this profile."
                )

                self.clear_recommendation()

                return

            # ---------------------------------------------
            # Save recommendation as PENDING
            # ---------------------------------------------

            recommendation_id = save_recommendation(
                patient_id=patient_id,
                assessment_id=assessment_id,
                exercise=recommendation["exercise"],
                difficulty=recommendation["difficulty"],
                reason=recommendation["reason"],
            )

            self.current_recommendation_id = (
                recommendation_id
            )

            # ---------------------------------------------
            # Display recommendation
            # ---------------------------------------------

            self.exercise_label.setText(
                recommendation["exercise"]
            )

            self.difficulty_label.setText(
                f"Difficulty: "
                f"{recommendation['difficulty']}"
            )

            self.reason_label.setText(
                f"Reason: {recommendation['reason']}"
            )

            self.reject_button.setEnabled(True)
            self.approve_button.setEnabled(True)

            QMessageBox.information(
                self,
                "Recommendation Generated",
                "Clinical assessment saved and an exercise "
                "recommendation has been generated.\n\n"
                "Please review the recommendation."
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Error",
                f"Could not process assessment:\n\n{error}"
            )

    # =====================================================
    # Approve recommendation
    # =====================================================

    def approve_recommendation(self):

        if self.current_recommendation_id is None:
            return

        try:

            update_recommendation_status(
                self.current_recommendation_id,
                "APPROVED"
            )

            self.reason_label.setText(
                "✓ Therapist approved this exercise. "
                "It is now assigned to the patient."
            )

            self.approve_button.setEnabled(False)
            self.reject_button.setEnabled(False)

            QMessageBox.information(
                self,
                "Exercise Approved",
                "The recommended exercise has been "
                "approved and assigned to the patient."
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Error",
                f"Could not approve recommendation:\n\n{error}"
            )

    # =====================================================
    # Reject recommendation
    # =====================================================

    def reject_recommendation(self):

        if self.current_recommendation_id is None:
            return

        try:

            update_recommendation_status(
                self.current_recommendation_id,
                "REJECTED"
            )

            self.reason_label.setText(
                "✕ Therapist rejected this recommendation. "
                "A different exercise can be selected later."
            )

            self.approve_button.setEnabled(False)
            self.reject_button.setEnabled(False)

        except Exception as error:

            QMessageBox.critical(
                self,
                "Error",
                f"Could not reject recommendation:\n\n{error}"
            )

    # =====================================================
    # Clear recommendation
    # =====================================================

    def clear_recommendation(self):

        self.current_recommendation_id = None

        self.exercise_label.setText(
            "No recommendation generated."
        )

        self.difficulty_label.setText(
            "Difficulty: —"
        )

        self.reason_label.setText(
            "No suitable exercise is currently available "
            "for the selected clinical profile."
        )

        self.approve_button.setEnabled(False)
        self.reject_button.setEnabled(False)

    # =====================================================
    # Reset form
    # =====================================================

    def clear_form(self, keep_patient=False):

        if not keep_patient:
            self.patient_input.setCurrentIndex(0)

        self.diagnosis_input.setCurrentIndex(0)
        self.impairment_input.setCurrentIndex(0)
        self.rom_input.setValue(0)
        self.restrictions_input.setCurrentIndex(0)
        self.notes_input.clear()

        self.clear_recommendation()