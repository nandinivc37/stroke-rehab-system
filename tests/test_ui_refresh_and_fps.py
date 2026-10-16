"""
Patient lists refresh after registration; the live frame rate is shown.

Offscreen Qt, temporary database, synthetic poses only.
"""

import pytest

import src.database.db as db
import src.ui.live_pose_widget as live
from src.adaptation.policy import PolicyConfig
from src.exercises.catalogue import get_exercise_config
from src.simulation.session_simulator import body_pose
from src.simulation.synthetic_pose import FakeCamera, ScriptedPose, make_results
from tests.helpers import prescribe, seed_patient


def items(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def register(window, monkeypatch, name="New Patient"):
    """Register through the real dialog's save_patient()."""
    from src.ui.pages import patients_page

    def fake_exec(dialog):
        dialog.name_input.setText(name)
        dialog.age_input.setValue(55)
        dialog.gender_input.setCurrentIndex(1)
        dialog.save_patient()
        return dialog.result()

    monkeypatch.setattr(patients_page.PatientDialog, "exec", fake_exec)
    window.patients_page.open_new_patient_dialog()
    return max(p[0] for p in db.get_all_patients())


# ---------------------------------------------------------
# Patient list refresh
# ---------------------------------------------------------

def test_registered_patient_appears_everywhere_without_restart(
    qapp, temp_db, monkeypatch
):
    from src.ui.main_window import MainWindow

    existing, _ = seed_patient()
    window = MainWindow()
    assessment = window.pages.widget(2)
    rehab = window.pages.widget(3)
    progress = window.pages.widget(4)
    assessment.patient_input.setCurrentIndex(
        assessment.patient_input.findData(existing))
    assessment.notes_input.setPlainText("half-filled form")

    new_id = register(window, monkeypatch)

    # No navigation and no restart: every list already has the patient.
    assert items(assessment.patient_input) == [None, existing, new_id]
    assert items(rehab.patient_combo) == [existing, new_id]
    assert items(progress.patient_combo) == [existing, new_id]
    assert window.patients_page.patient_table.rowCount() == 2

    # Existing selection and form contents are preserved.
    assert assessment.patient_input.currentData() == existing
    assert assessment.notes_input.toPlainText() == "half-filled form"


def test_repeated_refreshes_do_not_duplicate_patients(qapp, temp_db,
                                                      monkeypatch):
    from src.ui.main_window import MainWindow

    window = MainWindow()
    first = register(window, monkeypatch, "A")
    second = register(window, monkeypatch, "B")
    for index in range(window.pages.count()):
        window.show_page(index)
        window.show_page(index)
    assessment = window.pages.widget(2)
    assert items(assessment.patient_input) == [None, first, second]
    assert items(window.pages.widget(3).patient_combo) == [first, second]
    assert items(window.pages.widget(4).patient_combo) == [first, second]
    # Placeholder still selected: nothing was auto-selected.
    assert assessment.patient_input.currentIndex() == 0


def test_failed_registration_does_not_refresh(qapp, temp_db, monkeypatch):
    from src.ui.main_window import MainWindow
    from src.ui.pages import patients_page

    window = MainWindow()
    calls = []
    window.patients_page.patients_changed.connect(lambda: calls.append(1))

    def empty_name(dialog):
        dialog.save_patient()          # refuses: no name entered
        return dialog.result()

    monkeypatch.setattr(patients_page.PatientDialog, "exec", empty_name)
    window.patients_page.open_new_patient_dialog()
    assert calls == [] and db.get_all_patients() == []


def test_session_page_is_not_reloaded_mid_session(qapp, temp_db, monkeypatch,
                                                   scripted_camera):
    from src.ui.main_window import MainWindow

    patient, assessment_id = seed_patient()
    prescribe(patient, assessment_id, "Shoulder Abduction",
              "shoulder_abduction", "LEFT",
              {"target_reps": 3, "target_range": 80})
    window = MainWindow()
    rehab = window.pages.widget(3)
    rehab.patient_combo.setCurrentIndex(rehab.patient_combo.findData(patient))
    scripted_camera["results"] = [make_results()] * 3
    rehab.start_exercise()

    register(window, monkeypatch)
    assert items(rehab.patient_combo) == [patient]     # reloaded later
    assert rehab.current_patient_id == patient

    rehab.stop_exercise()
    window.show_page(3)
    assert len(items(rehab.patient_combo)) == 2
    assert rehab.current_patient_id == patient


# ---------------------------------------------------------
# Live frame rate
# ---------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def run_widget(monkeypatch, seconds_per_frame, frames=12):
    clock = Clock()
    monkeypatch.setattr(live.time, "monotonic", clock)
    pose = make_results(body_pose("shoulder_abduction", "LEFT", 40))
    widget = live.LivePoseWidget(get_exercise_config("shoulder_abduction"),
                                 "LEFT")
    widget.camera = FakeCamera(frames)
    widget.pose = ScriptedPose([pose] * frames)
    texts = []
    for _ in range(frames):
        clock.t += seconds_per_frame
        widget.update_frame()
        texts.append(widget.fps_label.text())
    return widget, texts


def test_measured_fps_below_gate_is_shown_with_warning(qapp, monkeypatch):
    widget, texts = run_widget(monkeypatch, 0.125)        # 8 fps
    assert texts[0] == "Frame rate: measuring…"
    text = texts[-1]
    assert "8.0 fps now" in text and "8.0 fps session average" in text
    assert "below 10 fps" in text and "cannot be assessed" in text
    assert live.FPS_LOW_COLOUR in widget.fps_label.styleSheet()
    # The displayed average is the value the gate checks.
    assert widget.get_stats()["mean_fps"] == pytest.approx(8.0)
    assert widget.analyzer.mean_fps == pytest.approx(8.0)


def test_measured_fps_above_gate_has_no_warning(qapp, monkeypatch):
    widget, texts = run_widget(monkeypatch, 0.05)         # 20 fps
    assert "20.0 fps now" in texts[-1]
    assert "20.0 fps session average" in texts[-1]
    assert "below" not in texts[-1]
    assert live.FPS_OK_COLOUR in widget.fps_label.styleSheet()


def test_fps_label_is_high_contrast_and_gate_unchanged(qapp):
    widget = live.LivePoseWidget(get_exercise_config("shoulder_abduction"),
                                 "LEFT")
    style = widget.fps_label.styleSheet()
    assert "background-color: #111827" in style
    assert "font-weight: 700" in style
    assert widget.min_fps == PolicyConfig().min_fps == 10.0
