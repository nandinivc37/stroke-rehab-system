"""
Patient registration, diagnosis editing and deletion safeguards.

Temporary database and synthetic sessions only; data/rehab.db is never
touched.
"""

import sqlite3

import pytest
from PyQt6.QtWidgets import QMessageBox

import src.database.db as db
from src.database.db import PATIENT_TABLES, PatientRecordError
from tests.helpers import perform, prescribe, seed_patient


def counts(patient_id):
    return db.get_patient_record_counts(patient_id)


def orphans():
    """Rows in patient tables whose patient no longer exists."""
    conn = db.get_connection()
    try:
        return {
            table: conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE patient_id NOT IN "
                "(SELECT patient_id FROM patients)").fetchone()[0]
            for table in PATIENT_TABLES
        }
    finally:
        conn.close()


@pytest.fixture
def two_patients_with_history(qapp, temp_db, scripted_camera):
    """Two patients, each with sessions, decisions and a proposal."""
    from src.ui.pages.rehab_page import RehabPage

    ids = []
    for _ in range(2):
        patient_id, assessment_id = seed_patient()
        prescribe(patient_id, assessment_id, "Shoulder Abduction",
                  "shoulder_abduction", "LEFT",
                  {"target_reps": 3, "target_range": 80})
        ids.append(patient_id)

    page = RehabPage()
    for patient_id in ids:
        page.patient_combo.setCurrentIndex(
            page.patient_combo.findData(patient_id))
        perform(page, scripted_camera, [84, 84, 84])
        perform(page, scripted_camera, [84, 84, 84])
    page.deleteLater()
    return ids


def select(page, patient_id):
    for row in range(page.patient_table.rowCount()):
        if page._row_patient_id(row) == patient_id:
            page.patient_table.selectRow(row)
            return
    raise AssertionError(f"patient {patient_id} not in table")


def items(combo):
    return [combo.itemData(i) for i in range(combo.count())]


# ---------------------------------------------------------
# Registration
# ---------------------------------------------------------

def test_registration_saves_patient_and_area_and_refreshes(
    qapp, temp_db, monkeypatch
):
    from src.ui.main_window import MainWindow
    from src.ui.pages import patients_page

    window = MainWindow()

    def fill(dialog):
        dialog.name_input.setText("Reg Test")
        dialog.age_input.setValue(61)
        dialog.gender_input.setCurrentIndex(1)
        dialog.side_input.setCurrentText("RIGHT")
        dialog.save_patient()
        return dialog.result()

    monkeypatch.setattr(patients_page.PatientDialog, "exec", fill)
    window.patients_page.open_new_patient_dialog()

    [(patient_id, name, age, *_)] = db.get_all_patients()
    assert (name, age) == ("Reg Test", 61)
    assert db.get_affected_areas(patient_id)[0][1] == "RIGHT"
    assert items(window.pages.widget(2).patient_input) == [None, patient_id]
    assert items(window.pages.widget(3).patient_combo) == [patient_id]
    # No assessment yet: diagnosis shown as missing, not invented.
    assert window.patients_page.patient_table.item(0, 7).text() == "—"


# ---------------------------------------------------------
# Diagnosis: database layer
# ---------------------------------------------------------

def test_diagnosis_update_changes_only_that_assessment(
    two_patients_with_history
):
    a, b = two_patients_with_history
    earlier = db.get_current_assessment(a)["assessment_id"]
    newest = db.save_clinical_assessment(a, "Stroke", "Reduced ROM", 30,
                                         "None", "newer")
    latest = db.get_current_assessment(a)
    assert latest["assessment_id"] == newest      # latest = most recent
    sessions_before = db.get_sessions(a)
    prescription_before = db.get_active_prescription(a)
    b_before = db.get_current_assessment(b)

    db.update_assessment_diagnosis(a, latest["assessment_id"],
                                   "Post-stroke hemiparesis")

    assert db.get_current_assessment(a)["diagnosis"] == \
        "Post-stroke hemiparesis"
    conn = db.get_connection()
    others = conn.execute(
        "SELECT assessment_id, diagnosis FROM clinical_assessments "
        "WHERE assessment_id != ?", (latest["assessment_id"],)).fetchall()
    conn.close()
    assert all(diagnosis == "Stroke" for _, diagnosis in others)
    assert earlier in [aid for aid, _ in others]   # history unchanged
    assert db.get_current_assessment(b) == b_before
    assert db.get_sessions(a) == sessions_before
    assert db.get_active_prescription(a) == prescription_before


@pytest.mark.parametrize("diagnosis", ["", "   ", None])
def test_empty_diagnosis_is_rejected(temp_db, diagnosis):
    patient_id, assessment_id = seed_patient()
    with pytest.raises(PatientRecordError, match="required"):
        db.update_assessment_diagnosis(patient_id, assessment_id, diagnosis)
    assert db.get_current_assessment(patient_id)["diagnosis"] == "Stroke"


def test_diagnosis_update_refuses_another_patients_assessment(temp_db):
    a, _ = seed_patient()
    b, b_assessment = seed_patient()
    with pytest.raises(PatientRecordError, match="does not belong"):
        db.update_assessment_diagnosis(a, b_assessment, "Other")
    assert db.get_current_assessment(b)["diagnosis"] == "Stroke"


# ---------------------------------------------------------
# Diagnosis: Patients page
# ---------------------------------------------------------

def edit_via_dialog(monkeypatch, choice):
    from src.ui.pages import patients_page
    seen = {}

    def fake_exec(dialog):
        seen["prefilled"] = dialog.diagnosis_input.currentText()
        dialog.diagnosis_input.setCurrentText(choice)
        dialog.save_diagnosis()
        return dialog.result()

    monkeypatch.setattr(patients_page.EditDiagnosisDialog, "exec", fake_exec)
    return seen


def test_edit_diagnosis_prefills_saves_and_refreshes(
    two_patients_with_history, monkeypatch, non_blocking_message_boxes
):
    from src.ui.main_window import MainWindow

    a, b = two_patients_with_history
    window = MainWindow()
    page = window.patients_page
    refreshed = []
    page.patients_changed.connect(lambda: refreshed.append(1))
    settings_before = db.get_active_prescription(a)["parameters"]

    select(page, a)
    seen = edit_via_dialog(monkeypatch, "Other")
    page.edit_selected_diagnosis()

    assert seen["prefilled"] == "Stroke"
    assert db.get_current_assessment(a)["diagnosis"] == "Other"
    assert db.get_current_assessment(b)["diagnosis"] == "Stroke"
    assert refreshed == [1]
    row = next(r for r in range(page.patient_table.rowCount())
               if page._row_patient_id(r) == a)
    assert page.patient_table.item(row, 7).text() == "Other"
    assert page.selected_patient_id() == a          # selection kept
    # The prescription is not silently changed, and the user is told.
    assert db.get_active_prescription(a)["parameters"] == settings_before
    _, title, text = non_blocking_message_boxes[-1]
    assert title == "Diagnosis Updated"
    assert "were not changed" in text and "not available" in text


def test_unchanged_or_unselected_diagnosis_is_not_saved(
    qapp, temp_db, monkeypatch, non_blocking_message_boxes
):
    from src.ui.pages.patients_page import PatientsPage

    patient_id, _ = seed_patient()
    page = PatientsPage()
    refreshed = []
    page.patients_changed.connect(lambda: refreshed.append(1))
    select(page, patient_id)

    edit_via_dialog(monkeypatch, "Stroke")             # same as before
    page.edit_selected_diagnosis()
    edit_via_dialog(monkeypatch, "Select diagnosis")   # placeholder
    page.edit_selected_diagnosis()

    assert refreshed == []
    assert db.get_current_assessment(patient_id)["diagnosis"] == "Stroke"
    titles = [title for _, title, _ in non_blocking_message_boxes]
    assert "No Change" in titles and "Missing Information" in titles


def test_edit_diagnosis_without_assessment_explains(
    qapp, temp_db, non_blocking_message_boxes
):
    from src.ui.pages.patients_page import PatientsPage

    patient_id = db.add_patient("No Assessment", 50, "Other", "T", "")
    page = PatientsPage()
    select(page, patient_id)
    page.edit_selected_diagnosis()
    assert non_blocking_message_boxes[-1][1] == "No Assessment"


def test_action_buttons_need_a_selection(qapp, temp_db):
    from src.ui.pages.patients_page import PatientsPage

    patient_id, _ = seed_patient()
    page = PatientsPage()
    assert not page.delete_patient_button.isEnabled()
    assert not page.edit_diagnosis_button.isEnabled()
    select(page, patient_id)
    assert page.delete_patient_button.isEnabled()
    assert page.edit_diagnosis_button.isEnabled()


# ---------------------------------------------------------
# Deletion: database layer
# ---------------------------------------------------------

def test_delete_removes_all_and_only_that_patients_records(
    two_patients_with_history, temp_db
):
    a, b = two_patients_with_history
    a_before, b_before = counts(a), counts(b)
    assert all(a_before.values()) and all(b_before.values())

    result = db.delete_patient(a)

    assert result["deleted"] == a_before
    assert db.get_patient(a) is None
    assert set(counts(a).values()) == {0}
    assert counts(b) == b_before
    assert db.get_patient(b) is not None
    assert set(orphans().values()) == {0}

    # The verified backup still holds the deleted patient's data.
    backup = sqlite3.connect(result["backup"])
    assert backup.execute("SELECT COUNT(*) FROM sessions WHERE patient_id=?",
                          (a,)).fetchone()[0] == a_before["sessions"]
    backup.close()
    assert result["backup"].parent == temp_db.parent / "backups"


def test_delete_refuses_when_other_patients_reference_the_records(temp_db):
    a, a_assessment = seed_patient()
    b, _ = seed_patient()
    # A (malformed) recommendation of B pointing at A's assessment.
    db.save_recommendation(b, a_assessment, "Shoulder Flexion", "Easy", "x")
    before = counts(a)

    with pytest.raises(PatientRecordError, match="other patients"):
        db.delete_patient(a)
    assert db.get_patient(a) is not None and counts(a) == before
    assert not (temp_db.parent / "backups").exists()   # nothing done


def test_delete_without_verified_backup_deletes_nothing(temp_db):
    from src.database.migrations import MigrationError

    patient_id, _ = seed_patient()
    journal = temp_db.with_name(temp_db.name + "-journal")
    journal.write_bytes(b"")          # backup refuses: not a clean snapshot
    with pytest.raises(MigrationError):
        db.delete_patient(patient_id)
    journal.unlink()
    assert db.get_patient(patient_id) is not None
    assert counts(patient_id)["clinical_assessments"] == 1


def test_delete_unknown_patient(temp_db):
    with pytest.raises(PatientRecordError, match="not found"):
        db.delete_patient(999)


# ---------------------------------------------------------
# Deletion: Patients page safeguards and refresh
# ---------------------------------------------------------

def answer(monkeypatch, button):
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: button)


def typed(monkeypatch, text, ok=True):
    from src.ui.pages import patients_page
    calls = []

    def get_text(*args, **kwargs):
        calls.append(args)
        return text, ok

    monkeypatch.setattr(patients_page.QInputDialog, "getText", get_text)
    return calls


def test_declining_confirmation_deletes_nothing(
    two_patients_with_history, monkeypatch
):
    from src.ui.pages.patients_page import PatientsPage

    a, _ = two_patients_with_history
    before = counts(a)
    page = PatientsPage()
    select(page, a)
    answer(monkeypatch, QMessageBox.StandardButton.No)
    page.delete_selected_patient()
    assert counts(a) == before and db.get_patient(a) is not None


def test_history_requires_typed_name(two_patients_with_history, monkeypatch,
                                     non_blocking_message_boxes):
    from src.ui.pages.patients_page import PatientsPage

    a, _ = two_patients_with_history
    page = PatientsPage()
    select(page, a)
    calls = typed(monkeypatch, "wrong name")
    page.delete_selected_patient()

    assert len(calls) == 1
    assert db.get_patient(a) is not None
    _, title, text = non_blocking_message_boxes[0]
    assert title == "Delete Patient"
    assert "2 rehabilitation session(s)" in text
    assert "cannot be undone" in text and "backup" in text
    assert non_blocking_message_boxes[-1][1] == "Not Deleted"


def test_confirmed_delete_refreshes_every_patient_list(
    two_patients_with_history, monkeypatch
):
    from src.ui.main_window import MainWindow

    a, b = two_patients_with_history
    window = MainWindow()
    assessment = window.pages.widget(2)
    rehab = window.pages.widget(3)
    progress = window.pages.widget(4)
    assessment.patient_input.setCurrentIndex(
        assessment.patient_input.findData(a))
    rehab.patient_combo.setCurrentIndex(rehab.patient_combo.findData(a))
    progress.patient_combo.setCurrentIndex(progress.patient_combo.findData(a))

    select(window.patients_page, a)
    typed(monkeypatch, "Synthetic Patient")
    window.patients_page.delete_selected_patient()

    assert db.get_patient(a) is None
    assert items(assessment.patient_input) == [None, b]
    assert items(rehab.patient_combo) == [b]
    assert items(progress.patient_combo) == [b]
    # No stale selection of the deleted patient anywhere.
    assert assessment.patient_input.currentData() is None
    assert assessment.current_recommendation_id is None
    assert rehab.current_patient_id == b
    assert progress.patient_combo.currentData() == b
    assert window.patients_page.patient_table.rowCount() == 1
    assert window.patients_page.selected_patient_id() is None


def test_patient_without_history_needs_only_one_confirmation(
    qapp, temp_db, monkeypatch
):
    from src.ui.pages.patients_page import PatientsPage

    patient_id = db.add_patient("Empty", 40, "Other", "T", "")
    page = PatientsPage()
    select(page, patient_id)
    calls = typed(monkeypatch, "unused")
    page.delete_selected_patient()
    assert calls == [] and db.get_patient(patient_id) is None


def test_deleting_last_patient_clears_session_page(qapp, temp_db,
                                                   monkeypatch):
    from src.ui.main_window import MainWindow

    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Shoulder Abduction",
              "shoulder_abduction", "LEFT", {"target_reps": 3})
    window = MainWindow()
    select(window.patients_page, patient_id)
    typed(monkeypatch, "Synthetic Patient")
    window.patients_page.delete_selected_patient()

    rehab = window.pages.widget(3)
    assert rehab.patient_combo.count() == 0
    assert rehab.current_patient_id is None
    assert not rehab.start_button.isEnabled()


def test_patient_in_running_session_cannot_be_deleted(
    qapp, temp_db, monkeypatch, scripted_camera, non_blocking_message_boxes
):
    from src.simulation.synthetic_pose import make_results
    from src.ui.main_window import MainWindow

    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Shoulder Abduction",
              "shoulder_abduction", "LEFT", {"target_reps": 3})
    window = MainWindow()
    rehab = window.pages.widget(3)
    rehab.patient_combo.setCurrentIndex(
        rehab.patient_combo.findData(patient_id))
    scripted_camera["results"] = [make_results()] * 3
    rehab.start_exercise()

    select(window.patients_page, patient_id)
    typed(monkeypatch, "Synthetic Patient")
    window.patients_page.delete_selected_patient()

    assert db.get_patient(patient_id) is not None
    assert non_blocking_message_boxes[-1][1] == "Session Running"
    rehab.stop_exercise()
