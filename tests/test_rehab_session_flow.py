"""
WS3/WS4: rehabilitation page workflow, offscreen, with a temporary DB.

The webcam and MediaPipe are replaced by scripted synthetic results, so
this verifies wiring and persistence only (webcam NOT TESTED).
"""

import json
import sqlite3


import src.database.db as db
from src.simulation.session_simulator import simulate_session
from tests.helpers import prescribe, seed_patient


def make_page():
    from src.ui.pages.rehab_page import RehabPage
    return RehabPage()


def run_session(page, frames):
    for _ in range(frames):
        page.live_widget.update_frame()
    page.stop_exercise()


def sessions(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM sessions")]
    conn.close()
    return rows


def test_prescribed_exercise_side_and_parameters_drive_session(
    qapp, temp_db, scripted_camera
):
    patient_id, assessment_id = seed_patient(side="RIGHT")
    rec_id = prescribe(
        patient_id, assessment_id, "Shoulder Abduction",
        "shoulder_abduction", "RIGHT",
        {"target_reps": 5, "target_range": 70},
    )

    results = simulate_session("shoulder_abduction", "RIGHT", [85] * 3)
    scripted_camera["results"] = results

    page = make_page()
    assert page.current_config.exercise_id == "shoulder_abduction"
    assert page.current_side == "RIGHT"
    assert page.current_parameters == {"target_reps": 5, "target_range": 70.0}
    assert "Target repetitions: 5" in page.prescription_label.text()

    page.start_exercise()
    widget = page.live_widget
    assert widget.exercise.exercise_id == "shoulder_abduction"
    assert widget.analyzer.up_threshold == 70.0
    # The displayed reference is the scoring reference.
    assert page.reference_widget.sequence is widget.analyzer.reference.frames

    run_session(page, len(results))

    [row] = sessions(temp_db)
    assert row["exercise"] == "Shoulder Abduction"
    assert row["exercise_id"] == "shoulder_abduction"
    assert row["affected_side"] == "RIGHT"
    assert row["prescription_id"] == rec_id
    assert row["metrics_version"] == db.METRICS_VERSION
    assert row["reps"] == 3
    assert row["target_reps"] == 5
    assert row["trajectory_similarity"] is not None
    assert row["valid_segments"] == 3
    assert row["frames_total"] == len(results)
    assert json.loads(row["parameters_json"]) == {
        "target_range": 70.0, "target_reps": 5
    }
    posture = json.loads(row["experimental_posture_json"])
    assert posture["experimental"] is True and posture["validated"] is False
    # No compensation classifier: its window-count columns stay NULL,
    # never 0. The rule-based check is stored as provisional/unvalidated.
    for column in ("comp_windows_total",
                   "comp_no_compensation", "comp_shoulder_elevation",
                   "comp_trunk_rotation", "comp_lean_forward"):
        assert row[column] is None
    assert row["compensation_method"] == "geometric_rules_v1"
    compensation = json.loads(row["compensation_json"])
    assert compensation["validated"] is False
    assert compensation["used_for_adaptation"] is False
    # Steady synthetic posture: nothing detected in what was assessed.
    assert compensation["any_detected"] is False
    assert "Session saved" in page.session_status.text()


def test_legacy_prescription_without_parameters_uses_defaults(
    qapp, temp_db, scripted_camera
):
    # Rows approved before the migration have only the exercise name.
    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Shoulder Flexion", None, None, None)

    page = make_page()
    assert page.current_config.exercise_id == "shoulder_flexion"
    assert page.current_side == "LEFT"  # from affected_areas
    assert page.current_parameters == (
        page.current_config.default_parameters()
    )
    assert "(default values)" in page.prescription_label.text()


def test_unsupported_approved_exercise_does_not_crash(
    qapp, temp_db, scripted_camera, non_blocking_message_boxes
):
    patient_id, assessment_id = seed_patient(joint="Wrist")
    prescribe(patient_id, assessment_id, "Wrist Flexion-Extension", None,
              None, None)

    page = make_page()
    assert page.current_config is None
    assert not page.start_button.isEnabled()
    assert "not supported" in page.session_status.text()

    page.start_exercise()  # guarded: shows a warning, starts nothing
    assert page.live_widget is None
    assert non_blocking_message_boxes[-1][0] == "warning"


def test_camera_failure_saves_nothing(qapp, temp_db, scripted_camera):
    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Elbow Flexion", "elbow_flexion",
              "LEFT", {})
    scripted_camera["open"] = False

    page = make_page()
    page.start_exercise()

    assert page.live_widget is None
    assert page.start_button.isEnabled()
    assert "No session was started" in page.session_status.text()
    assert sessions(temp_db) == []


def test_session_with_no_frames_is_not_saved(qapp, temp_db, scripted_camera):
    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Elbow Flexion", "elbow_flexion",
              "LEFT", {})
    scripted_camera["results"] = []

    page = make_page()
    page.start_exercise()
    page.stop_exercise()

    assert sessions(temp_db) == []
    assert "not saved" in page.session_status.text()


def test_unmeasured_session_stores_null_not_zero(qapp, temp_db,
                                                 scripted_camera):
    from src.simulation.synthetic_pose import empty_results

    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Elbow Flexion", "elbow_flexion",
              "LEFT", {})
    scripted_camera["results"] = [empty_results()] * 15

    page = make_page()
    page.start_exercise()
    run_session(page, 15)

    [row] = sessions(temp_db)
    assert row["reps"] == 0
    assert row["frames_total"] == 15
    assert row["frames_with_pose"] == 0
    assert row["max_angle"] is None
    assert row["min_angle"] is None
    assert row["trajectory_similarity"] is None
