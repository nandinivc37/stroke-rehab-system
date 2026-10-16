from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.assessment.recommendation import (
    DIAGNOSIS_OPTIONS,
    SUPPORTED_DIAGNOSES,
)
from src.database.db import (
    PatientRecordError,
    add_affected_area,
    add_patient,
    delete_patient,
    get_active_prescription,
    get_all_patients,
    get_affected_areas,
    get_current_assessment,
    get_patient,
    get_patient_record_counts,
    initialize_database,
    update_assessment_diagnosis,
)
from src.database.migrations import MigrationError

DANGER_BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        color: #b91c1c;
        border: 1px solid #b91c1c;
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 600;
    }
    QPushButton:hover { background-color: #fef2f2; }
    QPushButton:disabled { color: #6b7280; border-color: #d1d5db; }
"""


# Explicit colours keep dialogs readable under dark system themes.
DIALOG_STYLE = """
            QDialog {
                background-color: #ffffff;
            }

            QLabel {
                color: #374151;
            }

            QComboBox {
                color: #1f2937;
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 7px;
            }

            QComboBox:hover {
                border: 1px solid #2563eb;
            }

            QComboBox:focus {
                border: 1px solid #2563eb;
            }

            QComboBox QAbstractItemView {
                color: #1f2937;
                background-color: #ffffff;
                selection-color: #172554;
                selection-background-color: #dbeafe;
                border: 1px solid #d1d5db;
                outline: none;
            }

            QComboBox QAbstractItemView::item {
                color: #1f2937;
                background-color: #ffffff;
                padding: 8px;
            }

            QComboBox QAbstractItemView::item:hover {
                color: #172554;
                background-color: #dbeafe;
            }

            QComboBox QAbstractItemView::item:selected {
                color: #172554;
                background-color: #dbeafe;
            }
"""


class PatientDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet(DIALOG_STYLE)

        self.setWindowTitle("Register New Patient")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(15)

        title = QLabel("Register New Patient")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Enter patient information and the clinically identified affected area."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # -----------------------------------------
        # Patient information
        # -----------------------------------------

        form = QFormLayout()
        form.setSpacing(12)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Patient name")

        self.age_input = QSpinBox()
        self.age_input.setRange(0, 120)
        self.age_input.setValue(0)

        self.gender_input = QComboBox()
        self.gender_input.addItem("Select gender")
        self.gender_input.addItems([
            "Female",
            "Male",
            "Other",
            "Prefer not to say",
        ])

        self.gender_input.setCurrentIndex(0)

        self.therapist_input = QLineEdit()
        self.therapist_input.setPlaceholderText(
            "Therapist / Doctor"
        )

        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText(
            "General clinical notes..."
        )
        self.notes_input.setFixedHeight(80)

        form.addRow("Name:", self.name_input)
        form.addRow("Age:", self.age_input)
        form.addRow("Gender:", self.gender_input)
        form.addRow(
            "Therapist / Doctor:",
            self.therapist_input,
        )
        form.addRow("Notes:", self.notes_input)

        layout.addLayout(form)

        # -----------------------------------------
        # Affected area
        # -----------------------------------------

        area_title = QLabel("Affected Area")
        area_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #172554;"
        )

        layout.addWidget(area_title)

        area_form = QFormLayout()
        area_form.setSpacing(12)

        self.side_input = QComboBox()
        self.side_input.addItems([
            "LEFT",
            "RIGHT",
            "BILATERAL",
        ])

        self.body_region_input = QComboBox()
        self.body_region_input.addItems([
            "Upper Limb",
            "Lower Limb",
            "Trunk",
            "Full Body",
        ])

        self.joint_input = QComboBox()
        self.joint_input.addItems([
            "Shoulder",
            "Elbow",
            "Wrist",
            "Hip",
            "Knee",
            "Ankle",
            "Other",
        ])

        self.area_notes_input = QTextEdit()
        self.area_notes_input.setPlaceholderText(
            "Clinical notes about the affected area..."
        )
        self.area_notes_input.setFixedHeight(70)

        area_form.addRow(
            "Affected Side:",
            self.side_input,
        )
        area_form.addRow(
            "Body Region:",
            self.body_region_input,
        )
        area_form.addRow(
            "Joint:",
            self.joint_input,
        )
        area_form.addRow(
            "Clinical Notes:",
            self.area_notes_input,
        )

        layout.addLayout(area_form)

        # -----------------------------------------
        # Buttons
        # -----------------------------------------

        buttons = QHBoxLayout()
        buttons.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        save_button = QPushButton("Save Patient")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self.save_patient)

        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)

        layout.addLayout(buttons)

    def save_patient(self):

        name = self.name_input.text().strip()

        if not name:
            QMessageBox.warning(
                self,
                "Missing Information",
                "Please enter the patient's name.",
            )
            return

        age = self.age_input.value()
        gender = self.gender_input.currentText()
        therapist = (
            self.therapist_input.text().strip()
        )
        notes = self.notes_input.toPlainText().strip()

        side = self.side_input.currentText()
        body_region = (
            self.body_region_input.currentText()
        )
        joint = self.joint_input.currentText()
        area_notes = (
            self.area_notes_input.toPlainText().strip()
        )

        if self.gender_input.currentIndex() == 0:
            QMessageBox.warning(
                self,
                "Missing Information",
                "Please select the patient's gender.",
            )
            return

        if age == 0:
            QMessageBox.warning(
                self,
                "Missing Information",
                "Please enter the patient's age.",
            )
            return

        try:

            patient_id = add_patient(
                name,
                age,
                gender,
                therapist,
                notes,
            )

            add_affected_area(
                patient_id,
                side,
                body_region,
                joint,
                area_notes,
            )

            QMessageBox.information(
                self,
                "Patient Registered",
                f"Patient registered successfully.\n\n"
                f"Patient ID: {patient_id}",
            )

            self.accept()

        except Exception as error:

            QMessageBox.critical(
                self,
                "Database Error",
                f"Could not save patient:\n\n{error}",
            )


class EditDiagnosisDialog(QDialog):
    """
    Edit the diagnosis of a patient's latest clinical assessment.
    Prescriptions and exercise settings are not changed.
    """

    def __init__(self, patient_id, parent=None):
        super().__init__(parent)

        self.setStyleSheet(DIALOG_STYLE)
        self.setWindowTitle("Edit Diagnosis")
        self.setMinimumWidth(460)

        self.patient_id = patient_id
        self.patient = get_patient(patient_id)
        self.assessment = get_current_assessment(patient_id)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        areas = get_affected_areas(patient_id)
        area = (
            f"{areas[0][1]} {areas[0][3] or areas[0][2]}" if areas else "—"
        )
        form.addRow("Patient:", QLabel(
            f"{patient_id} - {self.patient['name']}"))
        form.addRow("Age / gender:", QLabel(
            f"{self.patient['age']} / {self.patient['gender']}"))
        form.addRow("Affected area:", QLabel(area))
        form.addRow("Assessment date:", QLabel(
            str(self.assessment["assessment_date"])))

        current = self.assessment["diagnosis"]
        self.diagnosis_input = QComboBox()
        self.diagnosis_input.addItem("Select diagnosis")
        self.diagnosis_input.addItems(DIAGNOSIS_OPTIONS)
        index = self.diagnosis_input.findText(current)
        self.diagnosis_input.setCurrentIndex(max(index, 0))
        form.addRow("Diagnosis:", self.diagnosis_input)
        layout.addLayout(form)

        note = QLabel(
            f"Current diagnosis: <b>{current}</b>. Only this patient's "
            "latest assessment is updated. Existing prescriptions, "
            "exercise settings and session history are not changed; "
            "record a new assessment for a new recommendation."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save Diagnosis")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save_diagnosis)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def save_diagnosis(self):

        if self.diagnosis_input.currentIndex() == 0:
            QMessageBox.warning(self, "Missing Information",
                                "Please select a diagnosis.")
            return

        diagnosis = self.diagnosis_input.currentText()

        if diagnosis == self.assessment["diagnosis"]:
            QMessageBox.information(self, "No Change",
                                    "The diagnosis is unchanged.")
            self.reject()
            return

        try:
            update_assessment_diagnosis(
                self.patient_id, self.assessment["assessment_id"], diagnosis
            )
        except PatientRecordError as error:
            QMessageBox.critical(self, "Not Saved", str(error))
            return

        message = f"Diagnosis updated to '{diagnosis}'."
        prescription = get_active_prescription(self.patient_id)
        if prescription is not None:
            message += (
                f"\n\nThe active prescription ({prescription['exercise']}) "
                "and its settings were not changed."
            )
            if diagnosis not in SUPPORTED_DIAGNOSES:
                message += (
                    " Exercise recommendations are not available for this "
                    "diagnosis; review the prescription on the Assessment "
                    "page."
                )
        QMessageBox.information(self, "Diagnosis Updated", message)
        self.accept()


class PatientsPage(QWidget):

    # Emitted after a patient was saved, so other pages reload their lists.
    patients_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        initialize_database()

        # Set by the main window: True while a session for that patient
        # is running (it must not be deleted mid-session).
        self.patient_in_session = lambda patient_id: False

        self.build_ui()
        self.load_patients()

    def build_ui(self):

        layout = QVBoxLayout(self)
        layout.setContentsMargins(45, 40, 45, 40)
        layout.setSpacing(20)

        # -----------------------------------------
        # Header
        # -----------------------------------------

        header = QHBoxLayout()

        title_area = QVBoxLayout()
        title_area.setSpacing(5)

        title = QLabel("Patients")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Manage patient profiles and affected areas"
        )
        subtitle.setObjectName("pageSubtitle")

        title_area.addWidget(title)
        title_area.addWidget(subtitle)

        header.addLayout(title_area)
        header.addStretch()

        new_patient_button = QPushButton(
            "+ New Patient"
        )
        new_patient_button.setObjectName(
            "primaryButton"
        )
        new_patient_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        new_patient_button.clicked.connect(
            self.open_new_patient_dialog
        )

        header.addWidget(new_patient_button)

        layout.addLayout(header)

        # Actions on the patient selected in the table.
        actions = QHBoxLayout()
        actions.addStretch()

        self.edit_diagnosis_button = QPushButton("Edit Diagnosis")
        self.edit_diagnosis_button.clicked.connect(
            self.edit_selected_diagnosis
        )
        self.delete_patient_button = QPushButton("Delete Patient")
        self.delete_patient_button.setStyleSheet(DANGER_BUTTON_STYLE)
        self.delete_patient_button.clicked.connect(
            self.delete_selected_patient
        )
        for button in (self.edit_diagnosis_button,
                       self.delete_patient_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setEnabled(False)
            actions.addWidget(button)

        layout.addLayout(actions)

        # -----------------------------------------
        # Patient table
        # -----------------------------------------

        self.patient_table = QTableWidget()

        self.patient_table.setColumnCount(8)

        self.patient_table.setHorizontalHeaderLabels([
            "Patient ID",
            "Name",
            "Age",
            "Gender",
            "Affected Side",
            "Body Region",
            "Joint",
            "Diagnosis",
        ])

        self.patient_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        self.patient_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )

        self.patient_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )

        self.patient_table.verticalHeader().setVisible(
            False
        )

        header = self.patient_table.horizontalHeader()

        header.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch
        )

        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
            3,
            QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
            4,
            QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
            5,
            QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
            6,
            QHeaderView.ResizeMode.Stretch
        )

        header.setSectionResizeMode(
            7,
            QHeaderView.ResizeMode.Stretch
        )

        self.patient_table.itemSelectionChanged.connect(
            self._selection_changed
        )

        layout.addWidget(self.patient_table)

    # ---------------------------------------------
    # Load patients
    # ---------------------------------------------

    def load_patients(self):

        selected = self.selected_patient_id()
        patients = get_all_patients()

        self.patient_table.clearSelection()
        self.patient_table.setRowCount(
            len(patients)
        )

        for row, patient in enumerate(patients):

            patient_id = patient[0]
            name = patient[1]
            age = patient[2]
            gender = patient[3]

            areas = get_affected_areas(
                patient_id
            )

            if areas:
                side = areas[0][1]
                body_region = areas[0][2]
                joint = areas[0][3]
            else:
                side = "--"
                body_region = "--"
                joint = "--"

            assessment = get_current_assessment(patient_id)

            values = [
                patient_id,
                name,
                age,
                gender,
                side,
                body_region,
                joint,
                assessment["diagnosis"] if assessment else "—",
            ]

            for column, value in enumerate(values):

                item = QTableWidgetItem(
                    str(value)
                )

                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )
                item.setData(Qt.ItemDataRole.UserRole, patient_id)

                self.patient_table.setItem(
                    row,
                    column,
                    item,
                )

        #self.patient_table.resizeColumnsToContents()

        # Keep the previous selection if that patient still exists.
        for row in range(self.patient_table.rowCount()):
            if self._row_patient_id(row) == selected:
                self.patient_table.selectRow(row)
                break
        self._selection_changed()

    def _row_patient_id(self, row):
        item = self.patient_table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def selected_patient_id(self):
        rows = self.patient_table.selectionModel().selectedRows()
        return self._row_patient_id(rows[0].row()) if rows else None

    def _selection_changed(self):
        selected = self.selected_patient_id() is not None
        self.edit_diagnosis_button.setEnabled(selected)
        self.delete_patient_button.setEnabled(selected)

    def _records_changed(self):
        self.load_patients()
        self.patients_changed.emit()

    # ---------------------------------------------
    # Edit diagnosis
    # ---------------------------------------------

    def edit_selected_diagnosis(self):

        patient_id = self.selected_patient_id()
        if patient_id is None:
            return

        if get_current_assessment(patient_id) is None:
            QMessageBox.information(
                self,
                "No Assessment",
                "This patient has no clinical assessment yet. Record one "
                "on the Assessment page to set a diagnosis.",
            )
            return

        dialog = EditDiagnosisDialog(patient_id, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._records_changed()

    # ---------------------------------------------
    # Delete patient
    # ---------------------------------------------

    def delete_selected_patient(self):

        patient_id = self.selected_patient_id()
        if patient_id is None:
            return

        patient = get_patient(patient_id)
        if patient is None:
            self._records_changed()
            return

        if self.patient_in_session(patient_id):
            QMessageBox.warning(
                self,
                "Session Running",
                "End the running rehabilitation session for this patient "
                "before deleting them.",
            )
            return

        counts = get_patient_record_counts(patient_id)
        label = f"{patient_id} - {patient['name']}"
        lines = [
            ("rehabilitation session(s) — the patient's progress history",
             counts["sessions"]),
            ("clinical assessment(s)", counts["clinical_assessments"]),
            ("exercise prescription(s) and recommendation(s)",
             counts["exercise_recommendations"]),
            ("adaptation decision(s)", counts["adaptation_decisions"]),
            ("affected-area record(s)", counts["affected_areas"]),
        ]
        details = "\n".join(f"  • {n} {text}" for text, n in lines)

        answer = QMessageBox.question(
            self,
            "Delete Patient",
            f"Permanently delete patient {label}?\n\n"
            f"This also deletes all of their records:\n{details}\n\n"
            "This cannot be undone in the application. A verified backup "
            "of the whole database is saved in data/backups/ first; the "
            "patient's data remains in that backup until it is deleted "
            "separately.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        if counts["sessions"] or counts["clinical_assessments"]:
            typed, ok = QInputDialog.getText(
                self,
                "Confirm Deletion of Clinical History",
                f"{counts['sessions']} session(s) and "
                f"{counts['clinical_assessments']} assessment(s) will be "
                f"deleted.\nType the patient's name ({patient['name']}) "
                "to confirm:",
            )
            if not ok or typed.strip() != patient["name"].strip():
                QMessageBox.information(
                    self, "Not Deleted",
                    "The name did not match. Nothing was deleted.",
                )
                return

        try:
            result = delete_patient(patient_id)
        except (PatientRecordError, MigrationError, OSError) as error:
            QMessageBox.critical(
                self, "Not Deleted",
                f"Patient {label} was not deleted:\n\n{error}",
            )
            return

        self.patient_table.clearSelection()
        self._records_changed()

        QMessageBox.information(
            self,
            "Patient Deleted",
            f"Patient {label} and their records were deleted.\n\n"
            f"Backup: {result['backup']}",
        )

    # ---------------------------------------------
    # New patient
    # ---------------------------------------------

    def open_new_patient_dialog(self):

        dialog = PatientDialog(self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._records_changed()