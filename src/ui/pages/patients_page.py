from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QDialog,
    QFormLayout,
    QHBoxLayout,
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

from src.database.db import (
    add_affected_area,
    add_patient,
    get_all_patients,
    get_affected_areas,
    initialize_database,
)


class PatientDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("""
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
        """)

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


class PatientsPage(QWidget):

    def __init__(self):
        super().__init__()

        initialize_database()

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

        # -----------------------------------------
        # Patient table
        # -----------------------------------------

        self.patient_table = QTableWidget()

        self.patient_table.setColumnCount(7)

        self.patient_table.setHorizontalHeaderLabels([
            "Patient ID",
            "Name",
            "Age",
            "Gender",
            "Affected Side",
            "Body Region",
            "Joint",
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

        layout.addWidget(self.patient_table)

    # ---------------------------------------------
    # Load patients
    # ---------------------------------------------

    def load_patients(self):

        patients = get_all_patients()

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

            values = [
                patient_id,
                name,
                age,
                gender,
                side,
                body_region,
                joint,
            ]

            for column, value in enumerate(values):

                item = QTableWidgetItem(
                    str(value)
                )

                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

                self.patient_table.setItem(
                    row,
                    column,
                    item,
                )

        #self.patient_table.resizeColumnsToContents()

    # ---------------------------------------------
    # New patient
    # ---------------------------------------------

    def open_new_patient_dialog(self):

        dialog = PatientDialog(self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.load_patients()