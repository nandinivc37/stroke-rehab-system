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
)

from src.database.db import (
    get_all_patients,
    get_affected_areas,
    save_clinical_assessment,
)


class AssessmentPage(QWidget):

    def __init__(self):
        super().__init__()

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
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(45, 40, 45, 40)
        layout.setSpacing(20)

        # -------------------------
        # Header
        # -------------------------

        title = QLabel("Clinical Assessment")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Enter patient-specific clinical information for exercise recommendation."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # -------------------------
        # Patient selection
        # -------------------------

        form = QFormLayout()
        form.setSpacing(14)

        self.patient_input = QComboBox()
        self.patient_input.addItem("Select patient", None)

        self.load_patients()

        self.patient_input.currentIndexChanged.connect(
            self.patient_changed
        )

        form.addRow("Patient:", self.patient_input)

        # -------------------------
        # Diagnosis
        # -------------------------

        self.diagnosis_input = QComboBox()
        self.diagnosis_input.addItem("Select diagnosis")
        self.diagnosis_input.addItems([
            "Stroke",
            "Post-stroke hemiparesis",
            "Other",
        ])

        form.addRow("Diagnosis:", self.diagnosis_input)

        # -------------------------
        # Impairment
        # -------------------------

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

        # -------------------------
        # Baseline ROM
        # -------------------------

        self.rom_input = QDoubleSpinBox()
        self.rom_input.setRange(0.0, 360.0)
        self.rom_input.setDecimals(1)
        self.rom_input.setValue(0.0)
        self.rom_input.setSuffix("°")

        form.addRow("Baseline ROM:", self.rom_input)

        # -------------------------
        # Restrictions
        # -------------------------

        self.restrictions_input = QComboBox()
        self.restrictions_input.addItem("Select restriction")
        self.restrictions_input.addItems([
            "No major restriction",
            "Limited range",
            "Pain-limited movement",
            "Therapist specified",
        ])

        form.addRow("Restrictions:", self.restrictions_input)

        # -------------------------
        # Clinical notes
        # -------------------------

        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText(
            "Additional clinical observations..."
        )
        self.notes_input.setFixedHeight(100)

        form.addRow("Clinical Notes:", self.notes_input)

        layout.addLayout(form)

        # -------------------------
        # Patient context
        # -------------------------

        self.area_label = QLabel(
            "Affected area: Select a patient"
        )
        self.area_label.setStyleSheet(
            "color: #1d4ed8; font-weight: bold;"
        )

        layout.addWidget(self.area_label)

        # -------------------------
        # Buttons
        # -------------------------

        buttons = QHBoxLayout()
        buttons.addStretch()

        clear_button = QPushButton("Reset Form")
        clear_button.clicked.connect(self.clear_form)

        save_button = QPushButton("Save Assessment")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self.save_assessment)

        buttons.addWidget(clear_button)
        buttons.addWidget(save_button)

        layout.addLayout(buttons)

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
                "Affected area: Select a patient"
            )
            return

        areas = get_affected_areas(patient_id)

        if not areas:
            self.area_label.setText(
                "Affected area: Not specified"
            )
            return

        area = areas[0]

        side = area[1]
        body_region = area[2]
        joint = area[3]

        self.area_label.setText(
            f"Affected area: {side} • {body_region} • {joint}"
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

        diagnosis = self.diagnosis_input.currentText()
        impairment = self.impairment_input.currentText()
        baseline_rom = self.rom_input.value()
        restrictions = self.restrictions_input.currentText()
        clinical_notes = self.notes_input.toPlainText().strip()

        try:

            assessment_id = save_clinical_assessment(
                patient_id,
                diagnosis,
                impairment,
                baseline_rom,
                restrictions,
                clinical_notes,
            )

            QMessageBox.information(
                self,
                "Assessment Saved",
                f"Clinical assessment saved successfully.\n\n"
                f"Assessment ID: {assessment_id}"
            )

            self.clear_form(keep_patient=True)

        except Exception as error:

            QMessageBox.critical(
                self,
                "Database Error",
                f"Could not save assessment:\n\n{error}"
            )

    # =====================================================
    # Clear form
    # =====================================================

    def clear_form(self, keep_patient=False):

        if not keep_patient:
            self.patient_input.setCurrentIndex(0)

        self.diagnosis_input.setCurrentIndex(0)
        self.impairment_input.setCurrentIndex(0)
        self.rom_input.setValue(0)
        self.restrictions_input.setCurrentIndex(0)
        self.notes_input.clear()