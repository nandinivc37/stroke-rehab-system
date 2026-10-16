"""
WS3: exercise catalogue, 3D angle measurements and the generic analyzer.

All inputs are synthetic poses from a rigid body model. Passing these
tests shows the software behaves as designed; it does NOT show that the
measurements are accurate on real webcam data (webcam NOT TESTED).
"""

import pytest

from src.assessment.recommendation import EXERCISE_LIBRARY, recommend_exercise
from src.exercises.analyzer import ExerciseAnalyzer
from src.exercises.angles import (
    world_elbow_flexion,
    world_forward_reach,
    world_humeral_elevation,
)
from src.exercises.catalogue import (
    CATALOGUE,
    get_exercise_config,
    implemented_exercises,
)
from src.simulation.session_simulator import (
    body_pose,
    simulate_session,
    timestamps,
)
from src.simulation.synthetic_pose import (
    LEFT_HIP,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_WRIST,
    SyntheticResults,
    make_results,
)

SIDES = ["LEFT", "RIGHT"]

# Landmark protobufs store float32, so exact angles carry ~1e-6 degree
# rounding error.
ANGLE_TOL = 1e-3
EXERCISE_IDS = [config.exercise_id for config in CATALOGUE]


def run(exercise_id, side, results, parameters=None, fps=30.0):
    analyzer = ExerciseAnalyzer(get_exercise_config(exercise_id), side,
                                parameters)
    for result, t in zip(results, timestamps(len(results), fps)):
        analyzer.update(result, t)
    return analyzer, analyzer.summary()


# ---------------------------------------------------------
# Angle measurements
# ---------------------------------------------------------

@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("value", [0, 30, 60, 90, 120, 150])
def test_flexion_elevation_and_plane(side, value):
    reading = world_humeral_elevation(
        make_results(body_pose("shoulder_flexion", side, value)), side
    )
    assert reading.value == pytest.approx(value, abs=ANGLE_TOL)
    if value >= 30:
        assert reading.plane == pytest.approx(0.0, abs=ANGLE_TOL)
    assert reading.reference_frame == "hips"


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("value", [30, 90, 150])
def test_abduction_elevation_and_plane(side, value):
    reading = world_humeral_elevation(
        make_results(body_pose("shoulder_abduction", side, value)), side
    )
    assert reading.value == pytest.approx(value, abs=ANGLE_TOL)
    assert reading.plane == pytest.approx(90.0, abs=ANGLE_TOL)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("value", [0, 45, 90, 130])
def test_elbow_flexion(side, value):
    reading = world_elbow_flexion(
        make_results(body_pose("elbow_flexion", side, value)), side
    )
    assert reading.value == pytest.approx(value, abs=ANGLE_TOL)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("value", [0, 30, 60, 90])
def test_forward_reach(side, value):
    reading = world_forward_reach(
        make_results(body_pose("forward_reach", side, value)), side
    )
    assert reading.value == pytest.approx(value, abs=1e-4)


def test_plane_is_unknown_when_arm_hangs():
    reading = world_humeral_elevation(
        make_results(body_pose("shoulder_flexion", "LEFT", 5)), "LEFT"
    )
    assert reading.plane is None


def test_hidden_hips_fall_back_to_camera_vertical():
    result = make_results(
        body_pose("shoulder_flexion", "LEFT", 90),
        visibility_overrides={LEFT_HIP: 0.1, RIGHT_HIP: 0.1},
    )
    reading = world_humeral_elevation(result, "LEFT")
    assert reading.reference_frame == "camera"
    assert reading.value == pytest.approx(90.0, abs=ANGLE_TOL)


def test_elevation_is_relative_to_trunk_when_hips_visible():
    # Leaning the trunk forward 20 degrees with the arm hanging along it
    # is not a shoulder movement.
    result = make_results(
        body_pose("shoulder_flexion", "LEFT", 0, trunk_forward_lean=20)
    )
    assert world_humeral_elevation(result, "LEFT").value == pytest.approx(
        0.0, abs=ANGLE_TOL
    )


def test_missing_world_landmarks_are_unmeasurable():
    result = make_results(body_pose("shoulder_flexion", "LEFT", 90))
    no_world = SyntheticResults(result.pose_landmarks, None)
    assert world_humeral_elevation(no_world, "LEFT") is None
    assert world_elbow_flexion(no_world, "LEFT") is None
    assert world_forward_reach(no_world, "LEFT") is None


def test_non_finite_world_landmarks_are_unmeasurable():
    result = make_results(body_pose("shoulder_flexion", "LEFT", 90))
    result.pose_world_landmarks.landmark[LEFT_WRIST].x = float("nan")
    assert world_elbow_flexion(result, "LEFT") is None


# ---------------------------------------------------------
# Analyzer behaviour per exercise
# ---------------------------------------------------------

def target(exercise_id):
    return get_exercise_config(exercise_id).default_parameters()[
        "target_range"
    ]


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
def test_full_repetitions_are_counted_and_scored(exercise_id, side):
    peak = target(exercise_id) + 10
    _, summary = run(exercise_id, side,
                     simulate_session(exercise_id, side, [peak] * 3))

    assert summary["reps"] == 3
    assert summary["off_plane_reps"] == 0
    assert summary["valid_segments"] == 3
    assert summary["trajectory_similarity"] is not None
    assert summary["trajectory_similarity"] > 80
    assert summary["max_angle"] == pytest.approx(peak, abs=0.5)
    assert summary["mean_fps"] == pytest.approx(30.0)
    assert summary["frame_errors"] == 0


@pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
def test_movements_below_target_are_not_counted(exercise_id):
    peak = target(exercise_id) - 10
    _, summary = run(exercise_id, "LEFT",
                     simulate_session(exercise_id, "LEFT", [peak] * 2))
    assert summary["reps"] == 0
    # The attempts are still recorded and scored as movement.
    assert summary["valid_segments"] == 2


@pytest.mark.parametrize("exercise_id", EXERCISE_IDS)
def test_target_parameter_sets_rep_threshold(exercise_id):
    config = get_exercise_config(exercise_id)
    parameter = config.parameter("target_range")
    low = parameter.minimum + 2 * parameter.step
    peak = low + parameter.step

    results = simulate_session(exercise_id, "LEFT", [peak] * 2)

    _, at_default = run(exercise_id, "LEFT", results)
    _, at_low = run(exercise_id, "LEFT", results, {"target_range": low})

    assert at_low["rep_threshold"] == low
    assert at_low["reps"] == 2
    if peak < parameter.default:
        assert at_default["reps"] == 0


def test_parameters_are_clamped_and_unknown_keys_dropped():
    config = get_exercise_config("shoulder_flexion")
    resolved = config.resolve_parameters(
        {"target_range": 999, "target_reps": 1.4, "bogus": 3}
    )
    assert resolved == {"target_range": 150.0, "target_reps": 3}


def test_sideways_raise_is_not_counted_as_flexion():
    results = simulate_session("shoulder_abduction", "LEFT", [110] * 2)
    _, summary = run("shoulder_flexion", "LEFT", results)
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 2
    assert all(abs(s["plane_at_peak"] - 90) < 1 for s in summary["segments"])


def test_forward_raise_is_not_counted_as_abduction():
    results = simulate_session("shoulder_flexion", "LEFT", [110] * 2)
    _, summary = run("shoulder_abduction", "LEFT", results)
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 2


def test_single_frame_glitches_are_not_scored_attempts():
    # Behaviour change vs the original implementation, deliberately:
    # isolated one-frame spikes no longer become scored "attempts".
    rest = make_results(body_pose("shoulder_abduction", "LEFT", 0))
    spike = make_results(body_pose("shoulder_abduction", "LEFT", 95))
    results = [rest] * 10 + ([spike] + [rest] * 9) * 4

    _, summary = run("shoulder_abduction", "LEFT", results)
    assert summary["valid_segments"] == 0
    assert summary["trajectory_similarity"] is None
    assert summary["reps"] == 0


def test_moderate_landmark_noise_keeps_rep_count():
    peak = target("shoulder_abduction") + 20
    results = simulate_session("shoulder_abduction", "LEFT", [peak] * 4,
                               noise_m=0.01, seed=3)
    _, summary = run("shoulder_abduction", "LEFT", results)
    assert summary["reps"] == 4


def test_dropped_frames_are_counted_not_measured():
    results = simulate_session("elbow_flexion", "RIGHT", [110] * 2,
                               drop_every=7)
    _, summary = run("elbow_flexion", "RIGHT", results)
    assert summary["frames_with_pose"] < summary["frames_total"]
    assert summary["frames_measured"] == summary["frames_with_pose"]
    assert summary["reps"] == 2


def test_occluded_arm_is_not_measured():
    hidden = make_results(
        body_pose("elbow_flexion", "RIGHT", 100),
        visibility_overrides={RIGHT_WRIST: 0.1},
    )
    _, summary = run("elbow_flexion", "RIGHT", [hidden] * 20)
    assert summary["frames_measured"] == 0
    assert summary["max_angle"] is None


def test_reference_curve_peaks_at_prescribed_target():
    for config in CATALOGUE:
        for side in SIDES:
            for value in (config.parameter("target_range").minimum,
                          config.parameter("target_range").default):
                reference = config.reference_fn(
                    side, config.resolve_parameters({"target_range": value})
                )
                assert max(reference.angle_curve) == pytest.approx(
                    value, abs=1e-3
                )
                assert reference.frames.shape == (30, 33, 2)
                assert reference.synthetic is True


# ---------------------------------------------------------
# Catalogue and recommendation
# ---------------------------------------------------------

def test_catalogue_ids_unique_and_configs_consistent():
    ids = [config.exercise_id for config in CATALOGUE]
    assert len(ids) == len(set(ids))

    for config in CATALOGUE:
        assert config.webcam_validated is False
        threshold = config.parameter(config.rep_threshold_parameter)
        # A rep must always be able to end below the up threshold.
        assert config.rep.down_threshold < threshold.minimum
        assert config.segmentation.end_threshold < (
            config.segmentation.start_threshold
        )


def test_lookup_by_id_name_and_legacy_name():
    assert get_exercise_config("shoulder_flexion").display_name == (
        "Shoulder Flexion"
    )
    assert get_exercise_config("shoulder abduction").exercise_id == (
        "shoulder_abduction"
    )
    assert get_exercise_config("Elbow Flexion-Extension").exercise_id == (
        "elbow_flexion"
    )
    assert get_exercise_config("Wrist Flexion-Extension") is None
    assert get_exercise_config(None) is None


def test_recommender_only_offers_implemented_exercises():
    implemented = {c.display_name for c in implemented_exercises()}
    assert set(EXERCISE_LIBRARY) == implemented


@pytest.mark.parametrize("joint,expected", [
    ("Shoulder", "shoulder_flexion"),
    ("Elbow", "elbow_flexion"),
    ("Wrist", None),
    ("Knee", None),
])
def test_recommendation_by_joint(joint, expected):
    region = "Lower Limb" if joint == "Knee" else "Upper Limb"
    result = recommend_exercise(
        diagnosis="Stroke", affected_side="LEFT", body_region=region,
        joint=joint, impairment="Reduced ROM",
    )
    if expected is None:
        assert result["success"] is False
    else:
        assert result["exercise_id"] == expected
