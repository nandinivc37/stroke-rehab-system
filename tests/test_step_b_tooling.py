"""
Step B preparation: recorder storage and external-evaluation windowing.

Synthetic recordings only. These tests check bookkeeping (labels, gaps,
visibility, resampling, annotations), not model accuracy.
"""

import numpy as np
import pytest

from src.compensation_detection.shared_features import FEATURE_NAMES
from src.compensation_detection.step_a_compare import OUTPUT_DIR, WINDOW
from src.compensation_detection.step_b_evaluate import (
    evaluate,
    read_annotations,
    recording_windows,
    valid_segments,
)
from src.data_collection.recorder import LandmarkRecorder, load_recording
from src.simulation.session_simulator import simulate_session
from src.simulation.synthetic_pose import LEFT_HIP, empty_results


def record(tmp_path, results, labels, fps=15.0, recording_id="S01_test",
           participant="S01"):
    recorder = LandmarkRecorder()
    for i, (result, label) in enumerate(zip(results, labels)):
        recorder.add_frame(result, i / fps, label)
    npz, _ = recorder.save(tmp_path, {
        "recording_id": recording_id, "participant": participant,
        "task": "forward_reach", "side": "LEFT", "operator": "OP1",
    })
    return load_recording(npz)


def session(n_reps=3, **kwargs):
    return simulate_session("forward_reach", "LEFT", [60] * n_reps, **kwargs)


def test_recorder_round_trip(tmp_path):
    results = session(1)
    rec = record(tmp_path, results, [1] * len(results))
    assert rec["world"].shape == (len(results), 33, 3)
    assert rec["metadata"]["video_saved"] is False
    assert rec["metadata"]["frames_detected"] == len(results)
    assert rec["metadata"]["mean_fps"] == pytest.approx(15.0)


def test_recorder_refuses_to_overwrite(tmp_path):
    results = session(1)
    record(tmp_path, results, [1] * len(results))
    with pytest.raises(FileExistsError):
        record(tmp_path, results, [1] * len(results))


def test_undetected_frames_are_stored_as_missing(tmp_path):
    results = session(1)
    results[5] = empty_results()
    rec = record(tmp_path, results, [1] * len(results))
    assert not rec["detected"][5]
    assert np.isnan(rec["world"][5]).all()


def test_windows_resampled_and_labelled(tmp_path):
    results = session(2)
    labels = [1] * len(results)
    rec = record(tmp_path, results, labels, fps=15.0)
    X, y = recording_windows(rec)
    # 15 fps -> 30 Hz roughly doubles the frame count.
    expected = 2 * (len(results) - 1) + 1 - WINDOW + 1
    assert X.shape == (expected, WINDOW, len(FEATURE_NAMES))
    assert set(y) == {1}


def test_unlabelled_and_mixed_windows_are_excluded(tmp_path):
    results = session(2)
    half = len(results) // 2
    labels = [0] * half + [4] * (len(results) - half)
    rec = record(tmp_path, results, labels)
    _, y = recording_windows(rec)
    assert set(y) == {4}


def test_gaps_and_hidden_joints_split_segments(tmp_path):
    results = session(2)
    results[30] = empty_results()
    from src.simulation.synthetic_pose import make_results, BodyPose
    results[60] = make_results(BodyPose(),
                               visibility_overrides={LEFT_HIP: 0.1})
    rec = record(tmp_path, results, [1] * len(results))
    segments = valid_segments(rec)
    assert len(segments) == 3
    assert all(30 not in s and 60 not in s for s in segments)


def test_reviewer_annotations_override_operator_labels(tmp_path):
    results = session(2)
    rec = record(tmp_path, results, [1] * len(results), fps=15.0)
    csv_path = tmp_path / "annotations.csv"
    duration = (len(results) - 1) / 15.0
    csv_path.write_text(
        "recording_id,start_s,end_s,label,annotator\n"
        f"S01_test,0,{duration / 2},2,R1\n"
    )
    _, y = recording_windows(rec, read_annotations(csv_path))
    assert set(y) == {2}  # unannotated remainder becomes unlabelled


def test_invalid_annotation_label_rejected(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "recording_id,start_s,end_s,label,annotator\nx,0,1,7,R1\n")
    with pytest.raises(ValueError):
        read_annotations(csv_path)


def test_evaluate_reports_baseline_and_ci():
    y = np.array([1, 1, 2, 2, 3, 3, 4, 4] * 3)
    groups = np.array(["A"] * 8 + ["B"] * 8 + ["C"] * 8)
    report = evaluate(None, y, groups, {
        "perfect": y.copy(),
        "majority": np.ones_like(y),
    })
    assert report["perfect"]["pooled_macro_f1"] == pytest.approx(1.0)
    assert report["majority"]["pooled_macro_f1"] < 0.3
    low, high = report["perfect"]["macro_f1_95ci_participant_bootstrap"]
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)


@pytest.mark.skipif(
    not (OUTPUT_DIR / "final_models_not_deployed"
         / "bilstm_shared_v1.pt").exists(),
    reason="Step A final models not trained yet",
)
def test_frozen_models_load_and_predict_shapes(tmp_path):
    import torch

    from src.compensation_detection.step_a_compare import (
        predict_bilstm,
        window_summary,
    )
    from src.compensation_detection.step_b_evaluate import load_frozen_models

    results = session(2)
    X, _ = recording_windows(record(tmp_path, results, [1] * len(results)))
    rf, (bilstm, mean, std) = load_frozen_models(
        OUTPUT_DIR / "final_models_not_deployed")
    rf_pred = rf.predict(window_summary(X))
    nn_pred = predict_bilstm(bilstm, X, mean, std, torch.device("cpu"))
    assert rf_pred.shape == nn_pred.shape == (len(X),)
    assert set(rf_pred) <= {1, 2, 3, 4} and set(nn_pred) <= {1, 2, 3, 4}
