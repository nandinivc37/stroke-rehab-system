"""
Compensation-model guards and the validation tooling.

  * the Kinect-trained checkpoint refuses MediaPipe features;
  * the live feature definitions differ from the training convention
    (established here on synthetic poses, not by measuring accuracy);
  * evaluate_sessions computes agreement metrics and verdicts from a
    database and ground-truth files (SYNTHETIC fixtures: these test the
    tool's arithmetic, they are not validation results);
  * the axis-convention checker's per-frame logic.
"""

import json
import sqlite3

import pytest

import src.database.db as db
from src.simulation.synthetic_pose import ArmPose, BodyPose, make_results
from src.validation import axis_convention
from src.validation.evaluate_sessions import auc, evaluate


# ---------------------------------------------------------
# Compensation model
# ---------------------------------------------------------

def test_checkpoint_refuses_mediapipe_features():
    pytest.importorskip("torch")
    from src.compensation_detection.live_features import FEATURE_NAMES
    from src.compensation_detection.lstm_inference import (
        CompensationLSTM,
        IncompatibleFeatures,
    )
    from src.paths import LSTM_MODEL_PATH

    if not LSTM_MODEL_PATH.exists():
        pytest.skip("original checkpoint not present")
    model = CompensationLSTM()
    with pytest.raises(IncompatibleFeatures):
        model.check_input_schema(FEATURE_NAMES, "mediapipe")
    model.check_input_schema(FEATURE_NAMES, "toronto_kinect")


def test_live_trunk_angle_uses_a_different_convention_from_training():
    """
    Upright trunk: Kinect training data (y up) gives ~0 deg between the
    trunk vector and +y; the live MediaPipe image-coordinate features
    (y down) give ~180 deg. Same feature name, different meaning.
    """
    from src.compensation_detection.live_features import (
        extract_live_features,
    )
    features = extract_live_features(make_results(BodyPose()))
    assert features["trunk_vertical_angle"] == pytest.approx(180, abs=1)


def test_application_never_reports_missing_compensation_as_none():
    from src.adaptation.controller import describe_compensation
    info = describe_compensation(None)
    assert info["status"] == "NOT_ASSESSED"
    assert info["used_in_decision"] is False


# ---------------------------------------------------------
# Validation tooling (synthetic fixtures)
# ---------------------------------------------------------

def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1], [1]) == 0.5
    assert auc([], [1]) is None


def make_db(path, rows, frames):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE sessions(session_id INTEGER, exercise_id
        TEXT, reps INTEGER, off_plane_reps INTEGER, trajectory_similarity
        REAL, dtw_similarity REAL, mean_fps REAL, frames_total INTEGER,
        frames_measured INTEGER)""")
    conn.execute("""CREATE TABLE session_frames(session_id INTEGER,
        frame_index INTEGER, t_s REAL, status TEXT, angle REAL)""")
    conn.executemany("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.executemany("INSERT INTO session_frames VALUES (?,?,?,?,?)", frames)
    conn.commit()
    conn.close()


def test_evaluate_sessions_metrics_and_verdicts(tmp_path):
    path = tmp_path / "copy.db"
    rows = [
        # id, exercise, reps, off, score, dtw, fps, total, measured
        (1, "shoulder_flexion", 5, 0, 90.0, 92.0, 15.0, 100, 95),
        (2, "shoulder_flexion", 4, 0, 85.0, 90.0, 15.0, 100, 95),
        (3, "shoulder_flexion", 5, 0, 88.0, 91.0, 15.0, 100, 95),
        (4, "shoulder_flexion", 0, 5, None, None, 15.0, 100, 95),
        (5, "shoulder_flexion", 0, 0, 60.0, 85.0, 15.0, 100, 95),
    ]
    frames = [(1, i, i * 0.1, "ok", 10.0 * i) for i in range(10)]
    make_db(path, rows, frames)
    labels = [
        {"session_id": "1", "manual_reps": "5", "movement_label": "correct"},
        {"session_id": "2", "manual_reps": "5", "movement_label": "correct"},
        {"session_id": "3", "manual_reps": "5", "movement_label": "correct"},
        {"session_id": "4", "manual_reps": "5",
         "movement_label": "off_plane"},
        {"session_id": "5", "manual_reps": "0",
         "movement_label": "incomplete"},
    ]
    gonio = [{"session_id": "1", "t_s": "0.5", "reference_value": "45"}]
    report = evaluate(path, labels, gonio)

    rc = report["rep_count"]["shoulder_flexion"]
    assert rc["sessions"] == 5
    assert rc["exact_match_rate"] == pytest.approx(0.6)
    assert rc["mean_abs_error"] == pytest.approx(1.2)

    angle = report["angle"]["shoulder_flexion"]
    # Frames at 0.3-0.7 s (angles 30-70) lie in the +/-0.25 s window.
    assert angle["readings"] == 1 and angle["bias"] == pytest.approx(5.0)

    assert report["plane"] == {"reps": 19, "accuracy": pytest.approx(19 / 19)}
    disc = report["score_discrimination"]
    assert disc["by_label"]["off_plane"]["sessions_without_score"] == 1
    assert disc["auc_movement_score_correct_vs_other"] == 1.0

    verdicts = report["verdicts"]
    assert verdicts["fps"] == "PASS"
    assert verdicts["rep_count:shoulder_flexion"] == "FAIL"
    assert verdicts["angle:shoulder_flexion"] == "INSUFFICIENT_DATA"
    assert verdicts["score_discrimination"] == "INSUFFICIENT_DATA"
    json.dumps(report)


def test_evaluate_sessions_reads_a_real_application_database(
        qapp, temp_db, scripted_camera):
    from src.ui.pages.rehab_page import RehabPage
    from tests.helpers import perform, prescribe, seed_patient

    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Shoulder Abduction",
              "shoulder_abduction", "LEFT",
              {"target_reps": 3, "target_range": 80})
    perform(RehabPage(), scripted_camera, [90, 90, 90])
    [row] = db.get_sessions(patient_id)
    report = evaluate(temp_db, [{"session_id": str(row["session_id"]),
                                 "manual_reps": "3",
                                 "movement_label": "correct"}])
    assert report["rep_count"]["shoulder_abduction"]["exact_match_rate"] == 1
    assert report["sessions"][0]["frames_logged"] == row["frames_total"]


# ---------------------------------------------------------
# Axis convention checker
# ---------------------------------------------------------

def test_axis_checker_accepts_the_assumed_convention():
    pose = BodyPose()
    pose.left_arm = ArmPose(flexion=30)
    results = make_results(pose)
    # Synthetic image x follows world x: the subject faces the camera.
    outcome = axis_convention.check_frame(results)
    assert outcome == {"facing": True, "x": True, "y": True,
                       "nose_z": True, "fwd": True}
    assert axis_convention.summarise([outcome])["all_consistent"] is True


def test_axis_checker_flags_a_flipped_depth_axis():
    results = make_results(BodyPose())
    for landmark in results.pose_world_landmarks.landmark:
        landmark.z = -landmark.z
    outcome = axis_convention.check_frame(results)
    assert outcome["nose_z"] is False
    assert axis_convention.summarise([outcome])["all_consistent"] is False
