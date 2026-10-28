"""
Metrics v2: DTW normalisation, time-based repetition counting, peak ROM
from in-plane movements, glitch and gap handling, per-frame log.

Synthetic poses only: these tests verify the software's definitions,
not measurement accuracy on webcam data.
"""

import math

import pytest

from src.exercises.analyzer import (
    STATUS_IMPLAUSIBLE,
    STATUS_NOT_VISIBLE,
    STATUS_OK,
    ExerciseAnalyzer,
    resample,
    rolling_median_peak,
)
from src.exercises.catalogue import get_exercise_config
from src.exercises.angles import LEFT_ELBOW, humeral_elevation
from src.quality_assessment.dtw import (
    dtw_alignment,
    dtw_distance,
    dtw_similarity,
)
from src.simulation.synthetic_pose import (
    ArmPose,
    BodyPose,
    make_results,
    world_points,
)

FLEXION = get_exercise_config("shoulder_flexion")
PARAMS = {"target_reps": 10, "target_range": 80.0}


# ---------------------------------------------------------
# DTW
# ---------------------------------------------------------

def test_dtw_identical_sequences_have_zero_error():
    seq = [0, 20, 40, 60, 40, 20, 0]
    assert dtw_distance(seq, seq, "path") == 0
    assert dtw_similarity(seq, seq, "path") == 100.0


def test_dtw_absorbs_timing_differences():
    # Same shape, performed slower: perfect alignment exists.
    assert dtw_alignment([0, 1, 2], [0, 0, 1, 1, 2, 2]) == (0.0, 6)


def test_path_normalisation_is_the_mean_matched_error():
    total, steps = dtw_alignment([0, 0], [5, 5])
    assert (total, steps) == (10.0, 2)
    assert dtw_distance([0, 0], [5, 5], "path") == 5.0
    # The original rule divides by n + m and halves the error.
    assert dtw_distance([0, 0], [5, 5], "n_plus_m") == 2.5


def test_path_normalisation_does_not_depend_on_sequence_length():
    reference = [0, 20, 40, 60, 80, 60, 40, 20, 0]
    offset = [v + 10 for v in reference]
    slow = [v for v in offset for _ in range(3)]
    assert dtw_distance(reference, offset, "path") == pytest.approx(10.0)
    assert dtw_distance(reference, slow, "path") == pytest.approx(10.0)
    assert dtw_distance(reference, offset, "n_plus_m") != pytest.approx(
        dtw_distance(reference, slow, "n_plus_m"))


def test_default_normalisation_is_the_original_rule():
    reference = [0, 40, 80, 40, 0]
    patient = [30, 35, 40, 35, 30]
    assert dtw_similarity(reference, patient) == dtw_similarity(
        reference, patient, "n_plus_m")


def test_corrected_score_is_lower_for_an_incomplete_movement():
    reference = [0, 20, 40, 60, 80, 60, 40, 20, 0]
    half = [25, 30, 35, 40, 35, 30, 25]
    assert dtw_similarity(reference, half, "path") < dtw_similarity(
        reference, half, "n_plus_m")


def test_unknown_normalisation_is_rejected():
    with pytest.raises(ValueError):
        dtw_distance([1], [1], "magic")


def test_empty_sequence_gives_zero_similarity_not_a_crash():
    assert dtw_similarity([], [1, 2], "path") == 0.0


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def flexion(value, **trunk):
    pose = BodyPose(**trunk)
    pose.left_arm = ArmPose(flexion=value)
    return make_results(pose)


def abduction(value):
    pose = BodyPose()
    pose.left_arm = ArmPose(abduction=value)
    return make_results(pose)


def run(samples, fps, config=FLEXION, params=PARAMS, start=0.0):
    """samples: list of (duration_s, value or callable(t) -> results)."""
    analyzer = ExerciseAnalyzer(config, "LEFT", params)
    t = start
    for duration, make in samples:
        t0 = t
        while t - t0 < duration - 1e-9:
            analyzer.update(make(t - t0), t)
            t += 1.0 / fps
    return analyzer


def repetition(hold, peak=90.0, kind=flexion):
    return [
        (1.0, lambda u: kind(10 + 70 * u)),
        (hold, lambda u: kind(peak)),
        (1.0, lambda u: kind(80 - 70 * u)),
        (1.0, lambda u: kind(10)),
    ]


# ---------------------------------------------------------
# Repetition counting
# ---------------------------------------------------------

@pytest.mark.parametrize("hold", [0.3, 0.5, 1.0])
def test_count_does_not_depend_on_frame_rate(hold):
    counts = {
        fps: run(repetition(hold) * 3, fps).reps
        for fps in (7, 10, 15, 30)
    }
    assert set(counts.values()) == {3}, counts


def test_frame_based_counting_was_frame_rate_dependent():
    """Documents the defect metrics v2 fixes (legacy-style config)."""
    from dataclasses import replace
    legacy_like = replace(
        FLEXION, rep=replace(FLEXION.rep, min_dwell_s=None, max_gap_s=None))
    assert run(repetition(0.3) * 3, 30, legacy_like).reps == 3
    assert run(repetition(0.3) * 3, 7, legacy_like).reps == 0


def test_single_frame_above_threshold_is_not_a_repetition():
    samples = [(1.0, lambda u: flexion(50))]
    analyzer = run(samples, 30)
    analyzer.update(flexion(85), 1.0)       # one frame over 80
    t = 1.0 + 1 / 30
    for _ in range(30):
        analyzer.update(flexion(10), t)
        t += 1 / 30
    assert analyzer.reps == 0


def test_partial_repetitions_are_not_counted():
    analyzer = run(repetition(1.0, peak=70) * 3, 30)
    assert analyzer.reps == 0
    assert analyzer.off_plane_reps == 0


def test_holding_at_the_top_does_not_add_repetitions():
    analyzer = run(repetition(5.0) * 2, 30)
    assert analyzer.reps == 2


def test_tracking_gap_abandons_the_repetition_in_progress():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", PARAMS)
    t = 0.0
    for value in [10, 40, 70, 90, 90, 90, 90, 90]:
        analyzer.update(flexion(value), t)
        t += 1 / 30
    assert analyzer.stage == "up"
    t += 2.0                                  # tracking lost for 2 s
    for _ in range(20):
        analyzer.update(flexion(10), t)
        t += 1 / 30
    assert analyzer.reps == 0
    assert analyzer.interrupted_reps == 1


def test_repetition_in_progress_at_session_end_is_not_counted():
    samples = [(1.0, lambda u: flexion(10 + 80 * u)), (1.0, lambda u: flexion(90))]
    summary = run(samples, 30).summary()
    assert summary["reps"] == 0
    assert summary["rep_in_progress_at_end"] is True


def test_off_plane_movement_is_tallied_not_counted():
    analyzer = run(repetition(1.0, kind=abduction) * 3, 30)
    assert analyzer.reps == 0
    assert analyzer.off_plane_reps == 3


# ---------------------------------------------------------
# Glitches, missing landmarks, invalid coordinates
# ---------------------------------------------------------

def test_implausible_jump_is_rejected_as_a_glitch():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", PARAMS)
    analyzer.update(flexion(20), 0.0)
    status = analyzer.update(flexion(170), 1 / 30).status   # 4500 deg/s
    assert status == STATUS_IMPLAUSIBLE
    assert analyzer.frames_implausible == 1
    assert analyzer.max_angle < 30
    # A lasting change is accepted from the next frame on.
    assert analyzer.update(flexion(170), 2 / 30).status == STATUS_OK


def test_low_visibility_frames_are_not_measured():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", PARAMS)
    hidden = make_results(visibility_overrides={LEFT_ELBOW: 0.2})
    assert analyzer.update(hidden, 0.0).status == STATUS_NOT_VISIBLE
    assert analyzer.frames_measured == 0


def test_non_finite_world_coordinates_are_unmeasurable():
    result = flexion(45)
    result.pose_world_landmarks.landmark[LEFT_ELBOW].x = math.nan
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", PARAMS)
    analyzer.update(result, 0.0)
    assert analyzer.frames_unmeasurable == 1
    assert analyzer.frames_measured == 0


def test_mirrored_input_breaks_the_flexion_plane():
    """
    Why MediaPipe must receive the UNMIRRORED frame: mirroring the scene
    flips the handedness of the trunk frame, so a forward raise reads as
    a raise behind the body (plane ~180 deg) and would be rejected.
    """
    pose = BodyPose()
    pose.left_arm = ArmPose(flexion=90)
    points = world_points(pose)
    elevation, plane = humeral_elevation(points, "LEFT")
    assert abs(plane) < 1
    mirrored = points.copy()
    mirrored[:, 0] *= -1
    _, mirrored_plane = humeral_elevation(mirrored, "LEFT")
    assert abs(abs(mirrored_plane) - 180) < 1


# ---------------------------------------------------------
# Peak ROM, segments, movement score
# ---------------------------------------------------------

def test_peak_rom_ignores_a_single_frame_spike():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", PARAMS)
    t = 0.0
    values = ([10] * 10 + list(range(10, 91, 5)) + [90] * 10 + [105]
              + [90] * 10 + list(range(90, 9, -5)) + [10] * 10)
    for value in values:
        analyzer.update(flexion(value), t)
        t += 1 / 30
    summary = analyzer.summary()
    assert summary["max_angle"] == pytest.approx(105, abs=0.5)
    assert summary["peak_rom"] == pytest.approx(90, abs=0.5)


def test_off_plane_movement_does_not_set_peak_rom_or_score():
    summary = run(repetition(1.0, peak=120, kind=abduction) * 3, 30).summary()
    assert summary["max_angle"] > 100
    assert summary["peak_rom"] is None
    assert summary["trajectory_similarity"] is None
    assert summary["off_plane_segments"] == 3
    assert summary["valid_segments"] == 0


def test_in_plane_movement_is_scored_with_separate_components():
    summary = run(repetition(1.0) * 2, 30).summary()
    assert summary["reps"] == 2
    assert summary["peak_rom"] == pytest.approx(90, abs=0.5)
    assert 0 <= summary["dtw_similarity"] <= 100
    assert summary["dtw_mean_error_deg"] >= 0
    assert summary["dtw_normalization"] == "path"
    for segment in summary["segments"]:
        expected = 0.5 * segment["dtw_similarity"] + 50 * segment[
            "range_completion"]
        assert segment["score"] == pytest.approx(expected, abs=0.01)


def test_trunk_lean_does_not_change_trunk_relative_measurement():
    """Angles are relative to the trunk: a 20 deg forward lean neither
    adds nor removes range (so lean must be caught by other means)."""
    def leaning(value):
        return flexion(value, trunk_forward_lean=20.0)

    plain = run(repetition(1.0) * 2, 30).summary()
    leant = run(repetition(1.0, kind=leaning) * 2, 30).summary()
    assert leant["reps"] == plain["reps"] == 2
    assert leant["peak_rom"] == pytest.approx(plain["peak_rom"], abs=0.5)


def test_smoothness_uses_a_fixed_time_base():
    values = [0.0, 10.0, 20.0, 30.0]
    times = [0.0, 0.1, 0.2, 0.3]
    assert resample(values, times, 20.0) == pytest.approx(
        [0, 5, 10, 15, 20, 25, 30])
    from src.exercises.analyzer import velocity_peak_count

    def both(fps):
        analyzer = run(repetition(1.0), fps)
        raw = velocity_peak_count(analyzer.movement_segments[0])
        return analyzer.summary()["segments"][0]["velocity_peaks"], raw

    v2_30, raw_30 = both(30)
    v2_10, raw_10 = both(10)
    # On the 10 Hz time base the same movement gives nearly the same
    # count at 10 and 30 fps; on raw frames it does not. (Below 10 fps
    # information is lost and the indicator stays frame-rate sensitive:
    # it remains exploratory.)
    assert abs(v2_30 - v2_10) <= 1
    assert abs(raw_30 - raw_10) > 5


def test_rolling_median_peak():
    assert rolling_median_peak([1, 2, 50, 3, 4]) == 4
    assert rolling_median_peak([7]) == 7
    assert rolling_median_peak([]) is None


# ---------------------------------------------------------
# Per-frame diagnostic log
# ---------------------------------------------------------

def test_every_frame_is_logged_with_its_status():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", PARAMS)
    analyzer.update(flexion(30), 10.0)
    analyzer.update(make_results(visibility_overrides={LEFT_ELBOW: 0.1}),
                    10.1)
    analyzer.update(None, 10.2)
    analyzer.record_error(10.3)
    log = analyzer.summary()["frame_log"]
    assert [entry[1] for entry in log] == [
        "ok", "not_visible", "no_pose", "error"]
    assert log[0][0] == 0.0 and log[3][0] == pytest.approx(0.3)
    assert log[0][2] == pytest.approx(30, abs=0.5)
    assert log[1][4] == pytest.approx(0.1)
    assert len(log) == analyzer.frames_total
