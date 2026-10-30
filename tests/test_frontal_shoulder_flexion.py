"""
Front-facing shoulder flexion: depth-robust elevation, explicit handling
of an unverifiable movement plane, and consistency of counting, DTW,
ROM and adaptation.

Depth errors are MODELLED: the arm's MediaPipe depth (z relative to the
shoulder) is scaled by a factor k. k = 1 is exact depth; k < 1 is a
compressed estimate; k <= 0 is a missing or wrong-signed one. Real
MediaPipe depth behaviour has NOT been measured; these tests show the
software behaves as designed under those assumptions, not that it is
accurate on webcam data (webcam NOT TESTED).
"""

from dataclasses import replace

import numpy as np
import pytest

from src.adaptation.policy import MAINTAIN
from src.exercises.analyzer import ExerciseAnalyzer
from src.exercises.angles import (
    PLANE_CALIBRATING,
    PLANE_DEPTH_AMBIGUOUS,
    PLANE_NEAR_VERTICAL,
    PLANE_OK,
    CalibratedHumeralElevation,
    world_humeral_elevation,
)
from src.exercises.catalogue import CATALOGUE, get_exercise_config
from src.exercises.reference import project_for_display
from src.simulation.session_simulator import repetition_values, timestamps
from src.simulation.synthetic_pose import (
    LEFT_HIP,
    RIGHT_HIP,
    ArmPose,
    BodyPose,
    SyntheticResults,
    make_results,
    world_points,
)
from tests.test_adaptation_policy import decide, session

FLEXION = get_exercise_config("shoulder_flexion")
SIDES = ["LEFT", "RIGHT"]
ARM = {"LEFT": (11, 13, 15), "RIGHT": (12, 14, 16)}
ANGLE_TOL = 1e-3


def frontal(side, flexion=0.0, abduction=0.0, k=1.0, elbow_scale=1.0,
            **overrides):
    """Seated patient facing the camera; arm depth scaled by k."""
    pose = BodyPose()
    arm = ArmPose(flexion=flexion, abduction=abduction)
    setattr(pose, "left_arm" if side == "LEFT" else "right_arm", arm)
    result = make_results(pose, **overrides)
    shoulder, elbow, wrist = ARM[side]
    world = result.pose_world_landmarks.landmark
    origin = np.array([world[shoulder].x, world[shoulder].y,
                       world[shoulder].z])
    for index in (elbow, wrist):
        p = np.array([world[index].x, world[index].y, world[index].z])
        offset = (p - origin) * elbow_scale
        offset[2] *= k
        world[index].x, world[index].y, world[index].z = origin + offset
    return result


def calibrated(side, k=1.0):
    measure = CalibratedHumeralElevation()
    for _ in range(measure.rest_frames):
        measure(frontal(side, k=k), side)
    assert measure.rest_length is not None
    return measure


def frontal_session(side, peaks, k=1.0, abduction=0.0):
    """One repetition per peak, each starting from rest."""
    frames = []
    for peak in peaks:
        for v in repetition_values(peak):
            frames.append(frontal(side, flexion=v,
                                  abduction=abduction if v > 0 else 0.0,
                                  k=k))
    return frames


def analyse(frames, side="LEFT", config=FLEXION, params=None):
    analyzer = ExerciseAnalyzer(config, side, params)
    for result, t in zip(frames, timestamps(len(frames))):
        analyzer.update(result, t)
    return analyzer, analyzer.summary()


# Behaviour before this change: stateless 3D elevation, unknown plane
# counted as valid.
UNCALIBRATED = replace(
    FLEXION, angle_fn_factory=None,
    plane_check=replace(FLEXION.plane_check, require_known_plane=False),
)


# ---------------------------------------------------------
# Measurement
# ---------------------------------------------------------

@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("flexion,abduction", [
    (0, 0), (30, 0), (60, 0), (90, 0), (120, 0), (150, 0),
    (0, 45), (0, 90), (60, 20),
])
def test_identical_to_3d_measure_when_depth_is_consistent(
        side, flexion, abduction):
    result = frontal(side, flexion, abduction)
    expected = world_humeral_elevation(result, side)
    reading = calibrated(side)(result, side)
    assert reading.value == pytest.approx(expected.value, abs=ANGLE_TOL)
    if expected.plane is None:
        assert reading.plane is None
        assert reading.plane_status == PLANE_NEAR_VERTICAL
    else:
        assert reading.plane == pytest.approx(expected.plane, abs=0.01)
        assert reading.plane_status == PLANE_OK
    assert reading.reference_frame == expected.reference_frame


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("k", [0.7, 0.5, 0.3])
def test_compressed_depth_biases_3d_measure_but_not_calibrated(side, k):
    """Root cause: a compressed depth estimate shortens the measured
    upper arm, so the 3D elevation reads low below 90 deg and high above
    it. The calibrated measure uses the resting arm length instead."""
    measure = calibrated(side, k)
    for true in (60, 75, 90, 105, 120):
        result = frontal(side, true, k=k)
        old = world_humeral_elevation(result, side).value
        new = measure(result, side)
        if true != 90:
            assert abs(old - true) > 3
        assert new.value == pytest.approx(true, abs=2.0)
        assert new.plane == pytest.approx(0.0, abs=1.0)
        assert new.plane_status == PLANE_OK


@pytest.mark.parametrize("true", [30, 45])
def test_low_angle_residual_bias_is_bounded_and_conservative(true):
    """The 5% length tolerance (rest jitter) costs some accuracy below
    about 50 deg under strong compression; it can only under-estimate."""
    measure = calibrated("LEFT", 0.3)
    value = measure(frontal("LEFT", true, k=0.3), "LEFT").value
    assert true - 6.0 <= value <= true + 0.5


def test_sideways_raise_still_reads_as_abduction_plane():
    measure = calibrated("LEFT", 0.5)
    reading = measure(frontal("LEFT", abduction=90, k=0.5), "LEFT")
    assert reading.plane == pytest.approx(90.0, abs=1.0)


@pytest.mark.parametrize("k", [0.0, 0.1])
def test_missing_depth_makes_the_plane_unknown_not_flexion(k):
    measure = calibrated("LEFT", k)
    reading = measure(frontal("LEFT", 90, k=k), "LEFT")
    assert reading.value == pytest.approx(90.0, abs=2.0)
    assert reading.plane is None
    assert reading.plane_status == PLANE_DEPTH_AMBIGUOUS


def test_depth_behind_the_body_is_not_forward():
    measure = calibrated("LEFT", 1.0)
    reading = measure(frontal("LEFT", 60, k=-0.5), "LEFT")
    assert abs(reading.plane) > 150
    assert reading.plane_status == PLANE_OK


def test_uncalibrated_readings_have_unknown_plane():
    measure = CalibratedHumeralElevation()
    reading = measure(frontal("LEFT", 90), "LEFT")
    assert reading.value == pytest.approx(90.0, abs=ANGLE_TOL)
    assert reading.plane is None
    assert reading.plane_status == PLANE_CALIBRATING
    assert measure.rest_length is None


def test_fidgeting_at_rest_does_not_calibrate():
    measure = CalibratedHumeralElevation()
    for i in range(30):
        measure(frontal("LEFT", elbow_scale=0.8 if i % 2 else 1.0), "LEFT")
    assert measure.rest_length is None


def test_raising_the_arm_interrupts_calibration():
    measure = CalibratedHumeralElevation()
    for value in [0, 0, 0, 0, 60, 0, 0, 0, 0]:
        measure(frontal("LEFT", value), "LEFT")
    assert measure.rest_length is None
    measure(frontal("LEFT", 0), "LEFT")
    assert measure.rest_length is not None


def test_hidden_hips_fall_back_to_camera_vertical():
    measure = calibrated("LEFT", 0.5)
    result = frontal("LEFT", 75, k=0.5,
                     visibility_overrides={LEFT_HIP: 0.1, RIGHT_HIP: 0.1})
    reading = measure(result, "LEFT")
    assert reading.reference_frame == "camera"
    assert reading.value == pytest.approx(75.0, abs=2.0)


def test_missing_world_landmarks_are_unmeasurable():
    measure = calibrated("LEFT")
    result = frontal("LEFT", 90)
    assert measure(SyntheticResults(result.pose_landmarks, None),
                   "LEFT") is None


# ---------------------------------------------------------
# Repetitions, ROM, DTW
# ---------------------------------------------------------

@pytest.mark.parametrize("side", SIDES)
def test_forward_raise_with_compressed_depth_is_counted(side):
    frames = frontal_session(side, [70] * 3, k=0.5)
    params = {"target_range": 60}
    _, before = analyse(frames, side, UNCALIBRATED, params)
    _, after = analyse(frames, side, FLEXION, params)

    assert before["reps"] == 0              # the reported failure
    assert after["reps"] == 3
    assert after["off_plane_reps"] == after["plane_unverified_reps"] == 0
    assert after["peak_rom"] == pytest.approx(70.0, abs=2.0)
    assert after["valid_segments"] == 3
    assert after["dtw_similarity"] is not None
    assert all(s["plane_verified"] for s in after["segments"])


def test_session_with_exact_depth_is_unchanged():
    frames = frontal_session("LEFT", [100] * 3)
    _, before = analyse(frames, "LEFT", UNCALIBRATED)
    _, after = analyse(frames, "LEFT", FLEXION)
    for key in ("reps", "off_plane_reps", "peak_rom", "valid_segments",
                "trajectory_similarity", "dtw_similarity",
                "dtw_mean_error_deg"):
        assert after[key] == before[key], key


def test_small_sideways_drift_with_compressed_depth_is_still_flexion():
    frames = frontal_session("LEFT", [100] * 3, k=0.3, abduction=15)
    _, before = analyse(frames, "LEFT", UNCALIBRATED)
    _, after = analyse(frames, "LEFT", FLEXION)
    assert before["reps"] == 0 and before["off_plane_reps"] == 3
    assert after["reps"] == 3


def test_sideways_raise_is_still_rejected_from_the_front():
    frames = [frontal("LEFT", abduction=v, k=0.5)
              for v in repetition_values(110) * 3]
    _, summary = analyse(frames)
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 3
    assert summary["plane_unverified_reps"] == 0


def test_unverifiable_plane_is_not_counted_scored_or_used_for_rom():
    frames = frontal_session("LEFT", [100] * 3, k=0.0)
    _, before = analyse(frames, "LEFT", UNCALIBRATED)
    _, summary = analyse(frames)

    assert before["reps"] == 3              # previously accepted blindly
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 3
    assert summary["plane_unverified_reps"] == 3
    assert summary["plane_unverified_segments"] == 3
    assert summary["valid_segments"] == 0
    assert summary["peak_rom"] is None
    assert summary["trajectory_similarity"] is None
    assert not any(s["plane_verified"] for s in summary["segments"])


def test_extension_behind_the_body_is_not_counted():
    frames = frontal_session("LEFT", [100] * 3, k=-0.5)
    _, summary = analyse(frames)
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 3
    assert summary["plane_unverified_reps"] == 0


def already_raised(peak=100):
    """A repetition that starts with the arm already up (no rest)."""
    return (list(np.linspace(40, peak, 15)) + [peak] * 10
            + list(np.linspace(peak, 0, 20)) + [0.0] * 10)


def test_repetition_before_calibration_is_unverified():
    values = already_raised() + repetition_values(100) * 2
    frames = [frontal("LEFT", v, k=0.5) for v in values]
    _, summary = analyse(frames)
    assert summary["reps"] == 2
    assert summary["plane_unverified_reps"] == 1


def test_rest_jitter_does_not_create_movements():
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(300):
        result = frontal("LEFT", k=0.5)
        for index in ARM["LEFT"][1:]:
            landmark = result.pose_world_landmarks.landmark[index]
            dx, dy, dz = rng.normal(0.0, 0.004, 3)
            landmark.x += dx
            landmark.y += dy
            landmark.z += dz
        frames.append(result)
    _, summary = analyse(frames)
    assert summary["max_angle"] < FLEXION.segmentation.start_threshold
    assert summary["valid_segments"] == 0
    assert summary["reps"] == 0


def test_each_session_calibrates_afresh():
    first = ExerciseAnalyzer(FLEXION, "LEFT")
    second = ExerciseAnalyzer(FLEXION, "LEFT")
    assert first.angle_fn is not second.angle_fn
    for result in frontal_session("LEFT", [90]):
        first.update(result, None)
    assert first.angle_fn.rest_length is not None
    first.reset()
    assert first.angle_fn.rest_length is None


@pytest.mark.parametrize("side", SIDES)
def test_reference_curve_matches_the_3d_definition(side):
    params = FLEXION.default_parameters()
    reference = FLEXION.reference_fn(side, params)
    poses = [
        make_results(BodyPose(**{
            ("left_arm" if side == "LEFT" else "right_arm"):
                ArmPose(flexion=v)
        }))
        for v in np.linspace(0, params["target_range"], 5)
    ]
    curve = [world_humeral_elevation(p, side).value for p in poses]
    assert max(reference.angle_curve) == pytest.approx(max(curve),
                                                       abs=ANGLE_TOL)
    assert reference.angle_curve[0] == pytest.approx(0.0, abs=ANGLE_TOL)


def test_other_exercises_keep_their_measurement():
    for config in CATALOGUE:
        if config.exercise_id == "shoulder_flexion":
            continue
        assert config.angle_fn_factory is None
        assert config.make_angle_fn() is config.angle_fn
        if config.plane_check is not None:
            assert not config.plane_check.require_known_plane


def test_oblique_reference_shows_the_forward_raise():
    rest = project_for_display(world_points(BodyPose()), "oblique")
    pose = BodyPose()
    pose.left_arm = ArmPose(flexion=90)
    raised = project_for_display(world_points(pose), "oblique")
    for frame in (rest, raised):
        assert np.all(np.isfinite(frame))
        assert frame.min() >= 0.0 and frame.max() <= 1.0
    # The wrist moves across the screen, not only up.
    assert abs(raised[15, 0] - rest[15, 0]) > 0.1
    assert FLEXION.reference_view == "oblique"
    frames = FLEXION.reference_fn("LEFT", FLEXION.default_parameters()).frames
    assert frames.shape[1:] == (33, 2)


# ---------------------------------------------------------
# Adaptation and live display
# ---------------------------------------------------------

def test_unverified_repetitions_block_progression():
    _, summary = analyse(frontal_session("LEFT", [100] * 5, k=0.0))
    unverified = summary["off_plane_reps"]
    assert unverified == summary["plane_unverified_reps"] == 5
    result = decide([session(1), session(2, off_plane=unverified)])
    assert result.decision == MAINTAIN


def test_live_widget_explains_calibration_and_counts_reps(qapp):
    import src.ui.live_pose_widget as live
    from src.simulation.synthetic_pose import FakeCamera, ScriptedPose

    frames = [frontal("LEFT", v, k=0.5)
              for v in already_raised() + repetition_values(100) * 2]
    widget = live.LivePoseWidget(FLEXION, "LEFT", threaded=False)
    widget.camera = FakeCamera(len(frames), fps=30)
    widget.pose = ScriptedPose(frames)
    texts = []
    for _ in frames:
        widget.update_frame()
        texts.append(widget.status_label.text())
    stats = widget.get_stats()

    assert "calibrate" in texts[0]
    assert "calibrate" not in texts[-1]
    _, alone = analyse(frames)
    assert stats["reps"] == alone["reps"] == 2
    assert stats["plane_unverified_reps"] == 1
    assert stats["dtw_similarity"] == alone["dtw_similarity"]


def test_small_attempt_below_plane_elevation_is_scored_as_before():
    # Never elevated enough (30 deg) for a plane: nothing to verify.
    frames = frontal_session("LEFT", [29] * 2)
    _, before = analyse(frames, "LEFT", UNCALIBRATED)
    _, after = analyse(frames)
    assert after["valid_segments"] == before["valid_segments"] == 2
    assert after["plane_unverified_segments"] == 0
