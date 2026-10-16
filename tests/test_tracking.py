"""
Arm-tracking percentage, required-landmark selection and affected-side
selection.

Synthetic poses only. They verify the software's bookkeeping and
landmark conventions as implemented; MediaPipe's real visibility
behaviour on webcam video is NOT tested here.
"""

import numpy as np
import pytest

import src.ui.live_pose_widget as live
from src.exercises.analyzer import ExerciseAnalyzer
from src.exercises.angles import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    arm_landmarks,
)
from src.exercises.catalogue import get_exercise_config
from src.simulation.session_simulator import (
    body_pose,
    simulate_session,
    timestamps,
)
from src.simulation.synthetic_pose import (
    FakeCamera,
    ScriptedPose,
    empty_results,
    make_results,
)

SIDES = ["LEFT", "RIGHT"]
EXERCISES = ["shoulder_flexion", "shoulder_abduction", "elbow_flexion",
             "forward_reach"]
HIPS = {LEFT_HIP, RIGHT_HIP}  # optional: camera-vertical fallback
UPPER_BODY = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
              LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP)


def analyzer_for(exercise_id, side):
    return ExerciseAnalyzer(get_exercise_config(exercise_id), side)


def feed(analyzer, results):
    for result, t in zip(results, timestamps(len(results))):
        analyzer.update(result, t)
    return analyzer.summary()


def hidden(exercise_id, side, value, *landmarks):
    return make_results(body_pose(exercise_id, side, value),
                        visibility_overrides={i: 0.1 for i in landmarks})


# ---------------------------------------------------------
# Tracking percentage
# ---------------------------------------------------------

def test_every_frame_is_accounted_for_once():
    analyzer = analyzer_for("shoulder_flexion", "LEFT")
    pose = body_pose("shoulder_flexion", "LEFT", 40)
    frames = (
        [make_results(pose)] * 10
        + [empty_results()] * 5
        + [hidden("shoulder_flexion", "LEFT", 40, LEFT_ELBOW)] * 4
        + [hidden("shoulder_flexion", "LEFT", 40, LEFT_ELBOW,
                  RIGHT_SHOULDER)] * 3
    )
    feed(analyzer, frames)
    analyzer.record_error()
    analyzer.record_error()

    tracking = analyzer.summary()["tracking"]
    assert tracking["frames_total"] == 24
    assert tracking["frames_measured"] == 10
    assert tracking["frames_no_pose"] == 5
    assert tracking["frames_not_visible"] == 7
    assert tracking["frame_errors"] == 2
    assert tracking["frames_unmeasurable"] == 0
    assert sum(tracking[k] for k in (
        "frames_measured", "frames_no_pose", "frames_not_visible",
        "frames_unmeasurable", "frame_errors")) == tracking["frames_total"]
    assert tracking["measured_fraction"] == pytest.approx(10 / 24)
    assert tracking["not_visible_by_landmark"] == {"left elbow": 7,
                                                   "right shoulder": 3}
    # Most frequent first.
    assert next(iter(tracking["not_visible_by_landmark"])) == "left elbow"


def test_visibility_exactly_at_threshold_is_not_tracked():
    analyzer = analyzer_for("shoulder_abduction", "LEFT")
    at = make_results(body_pose("shoulder_abduction", "LEFT", 40),
                      visibility_overrides={LEFT_ELBOW: 0.5})
    above = make_results(body_pose("shoulder_abduction", "LEFT", 40),
                         visibility_overrides={LEFT_ELBOW: 0.51})
    summary = feed(analyzer, [at, above])
    assert summary["frames_measured"] == 1


def test_no_frames_gives_no_fraction():
    analyzer = analyzer_for("shoulder_flexion", "LEFT")
    assert analyzer.measured_fraction is None
    assert analyzer.summary()["tracking"]["measured_fraction"] is None


# ---------------------------------------------------------
# Required landmarks match what each measurement uses
# ---------------------------------------------------------

def landmarks_used(exercise_id, side):
    """Landmarks whose world position changes the measured value."""
    config = get_exercise_config(exercise_id)
    base = make_results(body_pose(exercise_id, side, 60))
    reference = config.angle_fn(base, side).value
    used = set()
    for index in UPPER_BODY:
        moved = make_results(body_pose(exercise_id, side, 60))
        point = moved.pose_world_landmarks.landmark[index]
        point.x += 0.05
        point.y -= 0.04
        point.z += 0.03
        reading = config.angle_fn(moved, side)
        if reading is None or not np.isclose(reading.value, reference,
                                             atol=1e-6):
            used.add(index)
    return used


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("exercise_id", EXERCISES)
def test_required_landmarks_are_exactly_the_ones_measured(exercise_id, side):
    required = set(get_exercise_config(exercise_id).required_landmarks(side))
    used = landmarks_used(exercise_id, side)
    # Hips have a camera-vertical fallback and are deliberately optional.
    assert required == used - HIPS


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("exercise_id", ["shoulder_flexion",
                                         "shoulder_abduction"])
def test_hidden_wrist_does_not_reject_shoulder_elevation(exercise_id, side):
    _, elbow, wrist = arm_landmarks(side)
    summary = feed(analyzer_for(exercise_id, side), [
        hidden(exercise_id, side, 10, wrist),     # arm at rest, wrist cropped
        hidden(exercise_id, side, 10, elbow),
    ])
    assert summary["frames_measured"] == 1
    assert summary["tracking"]["not_visible_by_landmark"] == {
        f"{side.lower()} elbow": 1}


@pytest.mark.parametrize("exercise_id", ["elbow_flexion", "forward_reach"])
def test_wrist_still_required_where_it_is_measured(exercise_id):
    summary = feed(analyzer_for(exercise_id, "LEFT"),
                   [hidden(exercise_id, "LEFT", 40, LEFT_WRIST)])
    assert summary["frames_measured"] == 0


def test_resting_wrist_out_of_view_no_longer_fails_the_gate():
    """
    The suspected live failure: the wrist hangs outside a waist-height
    image crop whenever the arm is down. Repetitions are unaffected; the
    whole session was previously rejected as poorly tracked.
    """
    results = simulate_session("shoulder_flexion", "LEFT", [95] * 5)
    for result in results:
        world = result.pose_world_landmarks.landmark
        arm_down = world[LEFT_ELBOW].y - world[LEFT_SHOULDER].y > 0.2
        if arm_down:
            result.pose_landmarks.landmark[LEFT_WRIST].visibility = 0.05
    summary = feed(analyzer_for("shoulder_flexion", "LEFT"), results)
    assert summary["reps"] == 5
    assert summary["tracking"]["measured_fraction"] == 1.0


# ---------------------------------------------------------
# Affected-side selection
# ---------------------------------------------------------

@pytest.mark.parametrize("exercise_id", EXERCISES)
@pytest.mark.parametrize("moving", SIDES)
def test_only_the_affected_arm_is_measured(exercise_id, moving):
    other = "RIGHT" if moving == "LEFT" else "LEFT"
    peak = {"forward_reach": 80}.get(exercise_id, 110)
    results = simulate_session(exercise_id, moving, [peak] * 3)
    assert feed(analyzer_for(exercise_id, moving), results)["reps"] == 3
    assert feed(analyzer_for(exercise_id, other), results)["reps"] == 0


@pytest.mark.parametrize("side", SIDES)
def test_side_uses_that_sides_landmarks(side):
    shoulder, elbow, wrist = arm_landmarks(side)
    assert (shoulder, elbow, wrist) == (
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST) if side == "LEFT"
        else (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    )
    # The unaffected arm's elbow and wrist never gate the affected arm.
    other_elbow = RIGHT_ELBOW if side == "LEFT" else LEFT_ELBOW
    other_wrist = RIGHT_WRIST if side == "LEFT" else LEFT_WRIST
    summary = feed(analyzer_for("shoulder_abduction", side), [
        hidden("shoulder_abduction", side, 40, other_elbow, other_wrist)])
    assert summary["frames_measured"] == 1


# ---------------------------------------------------------
# Live status never shows a stale angle for an untracked frame
# ---------------------------------------------------------

def test_live_status_marks_untracked_frames(qapp):
    config = get_exercise_config("shoulder_abduction")
    frames = [make_results(body_pose("shoulder_abduction", "LEFT", 60)),
              hidden("shoulder_abduction", "LEFT", 60, LEFT_ELBOW)]
    widget = live.LivePoseWidget(exercise=config, affected_side="LEFT")
    widget.camera = FakeCamera(len(frames))
    widget.pose = ScriptedPose(frames)

    widget.update_frame()
    assert "60.0°" in widget.status_label.text()
    assert "Arm tracked: 100% of frames" in widget.status_label.text()

    widget.update_frame()
    text = widget.status_label.text()
    assert "not tracked" in text and "60.0°" not in text
    assert "Arm tracked: 50% of frames" in text
