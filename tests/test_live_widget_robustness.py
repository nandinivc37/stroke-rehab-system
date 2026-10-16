"""WS1: per-frame error handling, overlay mirroring, missing data."""

import pytest

import src.ui.live_pose_widget as live
from src.exercises.legacy import LEGACY_SHOULDER_RAISE_2D
from src.simulation.synthetic_pose import FakeCamera, ScriptedPose, empty_results
from tests.legacy_inputs import legacy_results


class ExplodingPose:
    def __init__(self, failures, then=None):
        self.failures = failures
        self.then = then or []
        self.calls = 0

    def process(self, _rgb):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("simulated pose failure")
        return self.then[self.calls - self.failures - 1]

    def close(self):
        pass


def make_widget(pose, frames):
    widget = live.LivePoseWidget(
        exercise=LEGACY_SHOULDER_RAISE_2D, affected_side="LEFT"
    )
    widget.camera = FakeCamera(frames)
    widget.pose = pose
    return widget


def test_frame_exception_does_not_propagate_and_is_reported(qapp):
    widget = make_widget(ExplodingPose(2, [legacy_results(0.0)]), 3)

    for _ in range(3):
        widget.update_frame()  # must not raise

    stats = widget.get_stats()
    assert stats["frame_errors"] == 2
    assert stats["frames_total"] == 3
    assert widget.consecutive_errors == 0  # reset by the good frame
    assert "simulated pose failure" in stats["last_error"]
    assert not stats["analysis_halted"]


def test_repeated_errors_halt_analysis_without_fake_output(qapp):
    n = live.MAX_CONSECUTIVE_ERRORS
    widget = make_widget(ExplodingPose(n + 5), n + 5)
    widget.timer.start(1000)

    for _ in range(n + 5):
        widget.update_frame()

    stats = widget.get_stats()
    assert stats["analysis_halted"]
    assert not widget.timer.isActive()
    assert stats["frame_errors"] == n  # halted: later frames not read
    assert "Analysis stopped" in widget.status_label.text()
    # Nothing was measured, so nothing is reported as a measurement.
    assert stats["reps"] == 0
    assert stats["max_angle"] is None
    assert stats["trajectory_similarity"] is None


def test_no_pose_session_reports_unavailable_not_zero(qapp):
    widget = make_widget(ScriptedPose([empty_results()] * 10), 10)
    for _ in range(10):
        widget.update_frame()

    stats = widget.get_stats()
    assert stats["frames_with_pose"] == 0
    assert stats["max_angle"] is None
    assert stats["min_angle"] is None
    assert stats["trajectory_similarity"] is None
    assert stats["valid_segments"] == 0


def test_overlay_text_is_drawn_after_mirroring(qapp, monkeypatch):
    calls = []
    real_flip = live.cv2.flip
    real_put = live.cv2.putText

    def flip(*args, **kwargs):
        calls.append("flip")
        return real_flip(*args, **kwargs)

    def put_text(*args, **kwargs):
        calls.append("text")
        return real_put(*args, **kwargs)

    monkeypatch.setattr(live.cv2, "flip", flip)
    monkeypatch.setattr(live.cv2, "putText", put_text)

    widget = make_widget(ScriptedPose([legacy_results(30.0)]), 1)
    widget.update_frame()

    assert "flip" in calls and "text" in calls
    assert calls.index("flip") < calls.index("text")
    assert calls.count("flip") == 1


def test_widget_does_not_load_bilstm():
    import inspect

    source = inspect.getsource(live)
    assert "CompensationLSTM(" not in source
    assert "lstm_inference" not in source


def test_camera_read_failure_is_reported(qapp):
    widget = make_widget(ScriptedPose([]), 0)
    for _ in range(live.MAX_CONSECUTIVE_READ_FAILURES):
        widget.update_frame()
    assert "not delivering frames" in widget.status_label.text()


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_invalid_side_rejected(side):
    from src.exercises.analyzer import ExerciseAnalyzer
    ExerciseAnalyzer(LEGACY_SHOULDER_RAISE_2D, side)
    with pytest.raises(ValueError):
        ExerciseAnalyzer(LEGACY_SHOULDER_RAISE_2D, "BILATERAL")


def test_standalone_prototype_keeps_bilstm_disabled():
    # src/pose_estimation/inference.py runs on import (opens a webcam),
    # so check its source rather than importing it.
    from src.paths import PROJECT_ROOT

    source = (PROJECT_ROOT / "src/pose_estimation/inference.py").read_text()
    assert "COMPENSATION_MODEL_ENABLED = False" in source
