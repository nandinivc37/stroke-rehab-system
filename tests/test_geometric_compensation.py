"""
Rule-based compensation detector (geometric_rules_v1): detection,
false positives, missing data, scale/side invariance, temporal logic,
persistence, live integration and adaptation safeguards.

Synthetic MediaPipe-style poses only. These tests verify the software's
geometry and state logic; they say nothing about accuracy on webcam
video or about clinical validity.
"""

import json

import numpy as np
import pytest

import src.database.db as db
from src.compensation_detection.geometric_detector import (
    DETECTED,
    NOT_DETECTED,
    SHOULDER_HIKING,
    TRUNK_DEPTH,
    TRUNK_IMAGE,
    UNABLE,
    DetectorConfig,
    GeometricCompensationDetector,
    feedback,
    measure,
)
from src.exercises.analyzer import ExerciseAnalyzer
from src.exercises.angles import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from src.exercises.catalogue import get_exercise_config
from src.simulation.session_simulator import (
    body_pose,
    simulate_session,
    timestamps,
)
from src.simulation.synthetic_pose import (
    empty_results,
    make_results,
    world_points,
)
from tests.helpers import prescribe, seed_patient

SHOULDER_WIDTH = 0.36           # synthetic body model


def pose(exercise="elbow_flexion", side="LEFT", value=0.0, **trunk):
    return make_results(body_pose(exercise, side, value, **trunk))


def run(detector, frames):
    """frames: list of (results, movement_active). Returns last output."""
    out = None
    for results, active in frames:
        out = detector.update(results, active)
    return out


def rest(n=20, **kwargs):
    return [(pose(**kwargs), False)] * n


def moving(n=20, **kwargs):
    return [(pose(value=60.0, **kwargs), True)] * n


# ---------------------------------------------------------
# Detection of each implemented pattern
# ---------------------------------------------------------

@pytest.mark.parametrize("perturbation, pattern", [
    (dict(trunk_lateral_lean=20.0), TRUNK_IMAGE),
    (dict(trunk_lateral_lean=-20.0), TRUNK_IMAGE),       # either side
    (dict(trunk_forward_lean=25.0), TRUNK_DEPTH),
    (dict(left_shoulder_raise=0.05), SHOULDER_HIKING),   # 0.14 widths
])
def test_each_pattern_is_detected(perturbation, pattern):
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    out = run(detector, rest() + moving(**perturbation))
    assert out[pattern].state == DETECTED
    assert out[pattern].threshold == detector.config.for_exercise(
        "elbow_flexion").patterns[[p.name for p in detector.config.patterns]
                                  .index(pattern)].enter_threshold
    summary = detector.summary()["patterns"][pattern]
    assert summary["episodes"] == 1 and summary["detected_fraction"] > 0


def test_trunk_lean_is_not_reported_as_shoulder_hiking():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    out = run(detector, rest() + moving(trunk_lateral_lean=20.0))
    assert out[TRUNK_IMAGE].state == DETECTED
    assert out[SHOULDER_HIKING].state == NOT_DETECTED
    assert abs(out[SHOULDER_HIKING].deviation) < 1e-6


def test_forward_lean_does_not_trigger_sideways_lean():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    out = run(detector, rest() + moving(trunk_forward_lean=25.0))
    assert out[TRUNK_DEPTH].state == DETECTED
    assert out[TRUNK_IMAGE].state == NOT_DETECTED


def test_backward_lean_and_shoulder_drop_are_not_flagged():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    out = run(detector, rest() + moving(trunk_forward_lean=-25.0,
                                        left_shoulder_raise=-0.05))
    assert out[TRUNK_DEPTH].state == NOT_DETECTED
    assert out[SHOULDER_HIKING].state == NOT_DETECTED


# ---------------------------------------------------------
# Normal movement: no false positives
# ---------------------------------------------------------

@pytest.mark.parametrize("exercise, peaks", [
    ("shoulder_flexion", [120] * 3),
    ("shoulder_abduction", [110] * 3),
    ("elbow_flexion", [120] * 3),
    ("forward_reach", [80] * 3),
    ("shoulder_abduction", [35] * 3),       # restricted range of motion
])
def test_normal_movement_is_not_flagged(exercise, peaks):
    results = (
        [make_results(body_pose(exercise, "LEFT", 0.0))] * 20
        + simulate_session(exercise, "LEFT", peaks, noise_m=0.004)
    )
    analyzer = ExerciseAnalyzer(get_exercise_config(exercise), "LEFT")
    detector = GeometricCompensationDetector("LEFT", exercise)
    for result, t in zip(results, timestamps(len(results))):
        analysis = analyzer.update(result, t)
        detector.update(result, analysis.movement_active)
    summary = detector.summary()
    assert summary["baseline_established"]
    assert summary["any_detected"] is False
    assert any(p["assessed"] for p in summary["patterns"].values())


def test_slow_movement_is_not_flagged():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    slow = [(pose(value=v), True)
            for v in np.linspace(0, 50, 300)]          # 10 s at 30 fps
    run(detector, rest() + slow)
    assert detector.summary()["any_detected"] is False


def test_landmark_jitter_below_threshold_is_not_flagged():
    rng = np.random.default_rng(1)
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    frames = rest()
    for _ in range(200):
        result = pose(value=60.0)
        for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
            landmark = result.pose_world_landmarks.landmark[index]
            landmark.x += rng.normal(0, 0.006)
            landmark.y += rng.normal(0, 0.006)
        frames.append((result, True))
    run(detector, frames)
    assert detector.summary()["any_detected"] is False


# ---------------------------------------------------------
# Missing, low-visibility and occluded landmarks
# ---------------------------------------------------------

def hidden(*indices, **kwargs):
    return make_results(body_pose("elbow_flexion", "LEFT", 60.0, **kwargs),
                        visibility_overrides={i: 0.1 for i in indices})


@pytest.mark.parametrize("frame, reason", [
    (empty_results(), "no pose detected"),
    (hidden(LEFT_HIP, RIGHT_HIP), "shoulders/hips not visible"),
    (hidden(RIGHT_SHOULDER), "shoulders/hips not visible"),
])
def test_missing_landmarks_are_unable_not_absent(frame, reason):
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    out = run(detector, rest() + [(frame, True)] * 30)
    for result in out.values():
        assert result.state == UNABLE and result.reason == reason
    summary = detector.summary()
    for p in summary["patterns"].values():
        assert p["assessed"] is False and p["detected_fraction"] is None
        assert p["frames"][NOT_DETECTED] == 0
    text, level = feedback(out, detector.config, True, True)
    assert "unable to assess" in text and "no compensation" not in text


def test_occluded_hips_during_a_lean_hide_nothing_as_negative():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    frames = rest() + [(hidden(LEFT_HIP, RIGHT_HIP, trunk_lateral_lean=25),
                        True)] * 30
    run(detector, frames)
    p = detector.summary()["patterns"][TRUNK_IMAGE]
    assert p["frames"] == {DETECTED: 0, NOT_DETECTED: 0, UNABLE: 30}


def test_no_baseline_means_unable():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    out = run(detector, moving(trunk_lateral_lean=25.0))   # never rested
    assert out[TRUNK_IMAGE].state == UNABLE
    assert out[TRUNK_IMAGE].reason == "no stable resting baseline"
    assert detector.summary()["baseline_established"] is False


def test_unstable_rest_does_not_become_a_baseline():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    fidgeting = [(pose(trunk_lateral_lean=a), False)
                 for a in [0, 12, -12] * 10]
    run(detector, fidgeting)
    assert detector.patterns[TRUNK_IMAGE].baseline is None
    # Once the patient sits still, calibration completes.
    run(detector, rest(20))
    assert detector.patterns[TRUNK_IMAGE].baseline == pytest.approx(0, abs=1e-6)


def test_resting_frames_are_not_assessed():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    out = run(detector, rest() + rest(trunk_lateral_lean=25.0))
    assert out[TRUNK_IMAGE].state == UNABLE
    assert out[TRUNK_IMAGE].reason == "resting"
    assert detector.summary()["patterns"][TRUNK_IMAGE]["assessed"] is False


# ---------------------------------------------------------
# Body scale and affected side
# ---------------------------------------------------------

@pytest.mark.parametrize("scale", [0.7, 1.0, 1.5])
def test_measurements_are_scale_invariant(scale):
    points = world_points(body_pose("elbow_flexion", "LEFT", 60.0,
                                    trunk_lateral_lean=12.0,
                                    left_shoulder_raise=0.04))
    reference = measure(points, "LEFT")
    scaled = measure(points * scale, "LEFT")
    for name in reference:
        assert scaled[name] == pytest.approx(reference[name], abs=1e-9)


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_hiking_is_measured_on_the_affected_side(side):
    raise_affected = {f"{side.lower()}_shoulder_raise": 0.05}
    other = "right" if side == "LEFT" else "left"
    raise_other = {f"{other}_shoulder_raise": 0.05}

    affected = GeometricCompensationDetector(side, "elbow_flexion")
    out = run(affected, rest() + moving(side=side, **raise_affected))
    assert out[SHOULDER_HIKING].state == DETECTED
    assert out[SHOULDER_HIKING].deviation == pytest.approx(
        0.05 / SHOULDER_WIDTH, rel=0.05)

    unaffected = GeometricCompensationDetector(side, "elbow_flexion")
    out = run(unaffected, rest() + moving(side=side, **raise_other))
    assert out[SHOULDER_HIKING].state == NOT_DETECTED


def test_shoulder_exercises_use_a_higher_hiking_threshold():
    elbow = GeometricCompensationDetector("LEFT", "elbow_flexion")
    shoulder = GeometricCompensationDetector("LEFT", "shoulder_abduction")
    hike = dict(left_shoulder_raise=0.045)                 # 0.125 widths
    assert run(elbow, rest() + moving(**hike))[SHOULDER_HIKING].state \
        == DETECTED
    assert run(shoulder, rest() + moving(**hike))[SHOULDER_HIKING].state \
        == NOT_DETECTED


# ---------------------------------------------------------
# Temporal smoothing and state transitions
# ---------------------------------------------------------

def lean(angle, active=True):
    return (pose(value=60.0, trunk_lateral_lean=angle), active)


def states(detector, frames, pattern=TRUNK_IMAGE):
    return [detector.update(r, a)[pattern].state for r, a in frames]


def test_single_frame_spikes_do_not_trigger():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    run(detector, rest())
    spikes = [lean(30) if i % 4 == 0 else lean(0) for i in range(60)]
    assert DETECTED not in states(detector, spikes)


def test_onset_needs_consecutive_frames_and_release_has_hysteresis():
    cfg = DetectorConfig()
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion", cfg)
    run(detector, rest())
    sequence = (
        [lean(15)] * 12        # above enter (10)
        + [lean(8.5)] * 12     # between exit (7) and enter: stays detected
        + [lean(2)] * 12       # below exit: released
    )
    seen = states(detector, sequence)
    first = seen.index(DETECTED)
    # Median smoothing (5) + onset (5 consecutive smoothed frames).
    assert cfg.onset_frames - 1 <= first <= cfg.onset_frames + \
        cfg.smoothing_frames
    assert all(s == DETECTED for s in seen[first:24])
    assert seen[-1] == NOT_DETECTED
    assert detector.summary()["patterns"][TRUNK_IMAGE]["episodes"] == 1


def test_brief_dropout_is_held_long_gap_is_unable():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    run(detector, rest() + [lean(20)] * 15)
    gap = [(empty_results(), True)]
    assert states(detector, gap * 3) == [DETECTED] * 3
    assert states(detector, gap)[0] == UNABLE
    # Episode state was reset: a new onset is needed.
    assert states(detector, [lean(20)])[0] == NOT_DETECTED


def test_episode_ends_when_movement_attempt_ends():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    run(detector, rest() + [lean(20)] * 15)
    assert states(detector, [lean(0, active=False)])[0] == UNABLE
    assert states(detector, [lean(20)])[0] == NOT_DETECTED


def test_feedback_text_never_claims_absence_for_unassessed_checks():
    detector = GeometricCompensationDetector("LEFT", "elbow_flexion")
    cfg = detector.config
    assert "calibrating" in feedback({}, cfg, True, False)[0]
    out = run(detector, rest() + moving())
    text, level = feedback(out, cfg, True, True)
    assert level == "ok" and text.endswith("no compensation detected")
    out = run(detector, [(hidden(LEFT_HIP, RIGHT_HIP), True)])
    text, _ = feedback(out, cfg, True, True)
    assert "no compensation detected" not in text
    out = run(detector, rest() + moving(trunk_lateral_lean=25))
    text, level = feedback(out, cfg, True, True)
    assert level == "alert" and "Keep your trunk upright" in text


# ---------------------------------------------------------
# Live pipeline, persistence and retrieval
# ---------------------------------------------------------

def session_frames(exercise, peaks, trunk=None):
    """20 resting frames (calibration), then repetitions."""
    return (
        [make_results(body_pose(exercise, "LEFT", 0.0))] * 20
        + simulate_session(exercise, "LEFT", peaks, trunk=trunk)
    )


def test_live_widget_shows_feedback_and_keeps_reps_and_dtw(qapp):
    import src.ui.live_pose_widget as live
    from src.simulation.synthetic_pose import FakeCamera, ScriptedPose

    exercise = get_exercise_config("shoulder_abduction")
    frames = session_frames("shoulder_abduction", [90] * 3,
                            trunk={"trunk_lateral_lean": 20.0})
    widget = live.LivePoseWidget(exercise, "LEFT", threaded=False)
    widget.camera = FakeCamera(len(frames), fps=30)
    widget.pose = ScriptedPose(frames)
    texts = []
    for _ in frames:
        widget.update_frame()
        texts.append(widget.compensation_label.text())
    stats = widget.get_stats()

    assert any("calibrating" in t for t in texts[:10])
    assert any("Keep your trunk upright" in t for t in texts)
    assert stats["compensation"]["patterns"][TRUNK_IMAGE]["episodes"] == 3

    # Repetition counting and DTW are identical to the analyzer alone.
    analyzer = ExerciseAnalyzer(exercise, "LEFT")
    for result, t in zip(frames, timestamps(len(frames))):
        analyzer.update(result, t)
    alone = analyzer.summary()
    assert stats["reps"] == alone["reps"] == 3
    assert stats["trajectory_similarity"] == alone["trajectory_similarity"]
    assert stats["dtw_similarity"] == alone["dtw_similarity"]


def test_detector_failure_does_not_stop_analysis(qapp, monkeypatch):
    import src.ui.live_pose_widget as live
    from src.simulation.synthetic_pose import FakeCamera, ScriptedPose

    frames = session_frames("shoulder_abduction", [90] * 2)
    widget = live.LivePoseWidget(get_exercise_config("shoulder_abduction"),
                                 "LEFT", threaded=False)
    widget.camera = FakeCamera(len(frames), fps=30)
    widget.pose = ScriptedPose(frames)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(widget.compensation, "update", broken)
    for _ in frames:
        widget.update_frame()
    stats = widget.get_stats()
    assert stats["reps"] == 2 and stats["frame_errors"] == 0
    assert "unable to assess (error: RuntimeError)" in \
        widget.compensation_label.text()


@pytest.fixture
def abduction_patient(temp_db):
    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Shoulder Abduction",
              "shoulder_abduction", "LEFT",
              {"target_reps": 3, "target_range": 80})
    return patient_id


def perform_frames(page, script, frames):
    script["results"] = frames
    page.start_exercise()
    for _ in frames:
        page.live_widget.update_frame()
    page.stop_exercise()
    return page.last_adaptation


def test_session_summary_is_persisted_and_shown(qapp, abduction_patient,
                                                scripted_camera):
    from src.ui.pages.progress_page import ProgressPage
    from src.ui.pages.rehab_page import RehabPage

    page = RehabPage()
    perform_frames(page, scripted_camera, session_frames(
        "shoulder_abduction", [84] * 3,
        trunk={"trunk_lateral_lean": 20.0}))
    assert "Compensation check" in page.session_status.text()
    assert "DETECTED" in page.session_status.text()

    [row] = db.get_sessions(abduction_patient)
    assert row["compensation_method"] == "geometric_rules_v1"
    stored = json.loads(row["compensation_json"])
    assert stored["patterns"][TRUNK_IMAGE]["episodes"] == 3
    assert stored["patterns"][SHOULDER_HIKING]["episodes"] == 0
    assert stored["validated"] is False

    progress = ProgressPage()
    progress.patient_combo.setCurrentIndex(
        progress.patient_combo.findData(abduction_patient))
    progress.table.selectRow(0)
    progress.session_selected()
    detail = progress.session_detail.text()
    assert "Compensation check (rule-based" in detail
    assert "Trunk lean (sideways in a front view): DETECTED" in detail
    assert "not clinically validated" in detail


def test_sessions_before_schema_3_show_no_compensation_text():
    from src.progress.history import compensation_text
    assert compensation_text({"compensation_json": None}) is None


def test_compensation_cannot_change_adaptation(qapp, temp_db,
                                               scripted_camera):
    """Same performance with and without detected compensation gives
    the same controller actions; changes still need approval."""
    from src.ui.pages.rehab_page import RehabPage
    from src.adaptation.workflow import pending_proposals

    actions = {}
    for label, trunk in (("clean", None),
                         ("leaning", {"trunk_lateral_lean": 20.0})):
        patient_id, assessment_id = seed_patient()
        prescribe(patient_id, assessment_id, "Shoulder Abduction",
                  "shoulder_abduction", "LEFT",
                  {"target_reps": 3, "target_range": 80})
        page = RehabPage()
        page.patient_combo.setCurrentIndex(
            page.patient_combo.findData(patient_id))
        outcomes = [perform_frames(page, scripted_camera, session_frames(
            "shoulder_abduction", [84] * 3, trunk=trunk)) for _ in range(2)]
        actions[label] = [(o["action"], o["decision"]) for o in outcomes]
        stored = json.loads(db.get_sessions(patient_id)[-1]
                            ["compensation_json"])
        assert stored["any_detected"] is (trunk is not None)
        assert len(pending_proposals(patient_id)) == 1   # awaits approval
    assert actions["clean"] == actions["leaning"]
    assert actions["clean"][-1] == ("PROPOSE_PROGRESSION", "INCREASE")
