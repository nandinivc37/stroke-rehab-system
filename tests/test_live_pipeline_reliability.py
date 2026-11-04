"""
Live-pipeline reliability: movement segmentation, repetition gap
handling and geometric compensation detection, and their interaction.

Every corrected behaviour here was first observed in recorded live
sessions (flexion.jsonl, 2026-11-01; not in the repository):
  * MediaPipe lost the whole pose for 1.1-2.5 s mid-raise; the 1 s gap
    rule abandoned repetitions whose arm was still raised on return;
  * the attempt signal (movement_active) stayed on through tracking
    loss and while a seated resting arm read 21-25°, above the 20° end
    threshold, starving the compensation baseline of resting frames;
  * MediaPipe world-landmark failures (trunk 150-179° from upright) were
    scored as compensation; the baseline was never refreshed.
The frames below are SYNTHETIC. They check the logic; they are not
evidence of webcam accuracy or clinical validity.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from src.compensation_detection.geometric_detector import (
    DETECTED,
    NOT_DETECTED,
    SHOULDER_HIKING,
    TRUNK_DEPTH,
    TRUNK_IMAGE,
    UNABLE,
    GeometricCompensationDetector,
)
from src.exercises.analyzer import STATUS_OK, ExerciseAnalyzer
from src.exercises.catalogue import (
    FLEXION_MAX_BRIDGED_GAP_S,
    get_exercise_config,
    implemented_exercises,
)
from src.simulation.session_simulator import body_pose
from src.simulation.synthetic_pose import empty_results, make_results

FLEXION = get_exercise_config("shoulder_flexion")
TARGET_60 = {"target_range": 60.0, "target_reps": 5}
SEATED_REST = 22.0      # resting elevation seen in recorded sessions


def flex(value, **trunk):
    return make_results(body_pose("shoulder_flexion", "LEFT", value,
                                  **trunk))


def hold(value, seconds, fps, **trunk):
    return [flex(value, **trunk)] * int(round(seconds * fps))


def ramp(start, end, seconds, fps, **trunk):
    return [flex(v, **trunk)
            for v in np.linspace(start, end, int(round(seconds * fps)))]


def lost(seconds, fps):
    return [empty_results()] * int(round(seconds * fps))


def raise_and_lower(peak, fps, rest=SEATED_REST, rise_s=1.0, hold_s=0.5,
                    pause_s=1.5, **trunk):
    return (ramp(rest, peak, rise_s, fps, **trunk)
            + hold(peak, hold_s, fps, **trunk)
            + ramp(peak, rest, rise_s, fps, **trunk)
            + hold(rest, pause_s, fps))


def run(frames, fps=30.0, config=FLEXION, detector=None):
    """Same wiring as the live widget. Returns (analyzer, per-frame
    FrameAnalysis list)."""
    analyzer = ExerciseAnalyzer(config, "LEFT", TARGET_60)
    analyses = []
    for index, result in enumerate(frames):
        analysis = analyzer.update(result, index / fps)
        if detector is not None:
            detector.update(
                result, analysis.movement_active,
                arm_at_rest=(analysis.status == STATUS_OK
                             and not analysis.movement_active),
            )
        analyses.append(analysis)
    return analyzer, analyses


def without(config, **segmentation):
    return replace(config, segmentation=replace(config.segmentation,
                                                **segmentation))


# ---------------------------------------------------------
# Movement segmentation (attempt signal)
# ---------------------------------------------------------

@pytest.mark.parametrize("fps", [14.0, 30.0])
def test_seated_rest_above_end_threshold_ends_the_attempt(fps):
    frames = (hold(SEATED_REST, 2.0, fps) + raise_and_lower(90, fps)
              + hold(SEATED_REST, 2.0, fps))
    analyzer, analyses = run(frames, fps)
    assert analyzer.reps == 1
    assert analyses[-1].movement_active is False
    assert analyzer.segment_end_reason.startswith("resting still")
    assert analyzer.resting_angle == pytest.approx(SEATED_REST, abs=1.0)
    assert analyzer.summary()["valid_segments"] == 1


def test_without_the_rest_rule_the_attempt_never_ends():
    """The live failure: resting at 22° never satisfies 'five frames at
    or below 20°', so the attempt signal stays on."""
    frames = (hold(SEATED_REST, 2.0, 30) + raise_and_lower(90, 30)
              + hold(SEATED_REST, 5.0, 30))
    _, analyses = run(frames, config=without(FLEXION, rest_end_s=None))
    assert analyses[-1].movement_active is True


def test_still_hold_above_start_threshold_is_not_rest():
    """A restricted patient holding a small raise above the start
    threshold is still attempting the movement."""
    frames = hold(SEATED_REST, 2.0, 30) + ramp(SEATED_REST, 35, 1.0, 30) \
        + hold(35, 4.0, 30)
    _, analyses = run(frames)
    assert analyses[-1].movement_active is True


@pytest.mark.parametrize("fps", [14.0, 30.0])
def test_attempt_ends_while_tracking_is_lost(fps):
    frames = (hold(SEATED_REST, 2.0, fps) + ramp(SEATED_REST, 90, 1.0, fps)
              + lost(4.0, fps))
    analyzer, analyses = run(frames, fps)
    assert analyses[-1].movement_active is False   # before tracking returns
    assert analyzer.abandoned_segments == 1
    assert analyzer.segment_end_reason.startswith("abandoned")
    assert analyzer.summary()["valid_segments"] == 0   # not scored


def test_other_exercises_keep_their_configuration():
    for config in implemented_exercises():
        if config.exercise_id == "shoulder_flexion":
            continue
        assert config.segmentation.rest_end_s is None, config.exercise_id
        assert config.segmentation.max_gap_s is None, config.exercise_id
        assert config.rep.max_bridged_gap_s is None, config.exercise_id


# ---------------------------------------------------------
# Repetition state machine with tracking dropouts
# ---------------------------------------------------------

def dropout_rep(gap_s, return_angle, fps):
    return (hold(SEATED_REST, 2.0, fps) + ramp(SEATED_REST, 90, 1.0, fps)
            + hold(90, 0.5, fps) + lost(gap_s, fps)
            + hold(return_angle, 0.5, fps)
            + ramp(return_angle, SEATED_REST, 1.0, fps)
            + hold(SEATED_REST, 2.0, fps))


# Tracking returns with the arm at 50°: above the 30° lowering threshold
# but below the 60° target. Before bridging, such a repetition was lost
# (a return ABOVE the target simply starts a new raised stage).
RETURN_BELOW_TARGET = 50.0


@pytest.mark.parametrize("fps", [14.0, 30.0])
@pytest.mark.parametrize("gap_s", [1.5, 2.5])
def test_short_dropout_with_arm_still_raised_is_bridged(fps, gap_s):
    analyzer, _ = run(dropout_rep(gap_s, RETURN_BELOW_TARGET, fps), fps)
    assert analyzer.reps == 1
    assert analyzer.bridged_gaps == 1
    assert analyzer.interrupted_reps == 0


def test_without_bridging_the_same_repetition_is_lost():
    config = replace(FLEXION, rep=replace(FLEXION.rep,
                                          max_bridged_gap_s=None))
    analyzer, _ = run(dropout_rep(2.0, RETURN_BELOW_TARGET, 30),
                      config=config)
    assert analyzer.reps == 0
    assert analyzer.interrupted_reps == 1


def test_dropout_after_which_the_arm_is_down_is_interrupted():
    """The lowering was not observed: not credited."""
    analyzer, _ = run(dropout_rep(2.0, SEATED_REST, 30))
    assert analyzer.reps == 0
    assert analyzer.interrupted_reps == 1
    assert analyzer.bridged_gaps == 0


def test_dropout_longer_than_the_bridge_is_interrupted():
    analyzer, _ = run(dropout_rep(FLEXION_MAX_BRIDGED_GAP_S + 0.5,
                                  RETURN_BELOW_TARGET, 30))
    assert analyzer.reps == 0
    assert analyzer.interrupted_reps == 1


def test_bridging_does_not_count_without_the_lowering():
    frames = (hold(SEATED_REST, 2.0, 30) + ramp(SEATED_REST, 90, 1.0, 30)
              + lost(2.0, 30) + hold(RETURN_BELOW_TARGET, 2.0, 30))
    analyzer, _ = run(frames)
    assert analyzer.reps == 0
    assert analyzer.stage == "up"


@pytest.mark.parametrize("fps", [14.0, 30.0])
def test_slow_repetition_counts(fps):
    frames = (hold(SEATED_REST, 2.0, fps)
              + raise_and_lower(90, fps, rise_s=8.0, hold_s=2.0))
    analyzer, _ = run(frames, fps)
    assert analyzer.reps == 1
    assert analyzer.summary()["valid_segments"] == 1


def test_incomplete_raise_is_not_counted_but_is_recorded():
    frames = hold(SEATED_REST, 2.0, 30) + raise_and_lower(50, 30)
    analyzer, _ = run(frames)
    assert analyzer.reps == 0
    assert analyzer.off_plane_reps == 0
    assert analyzer.summary()["valid_segments"] == 1


def test_sideways_abduction_is_not_counted_as_flexion():
    frames = [make_results(body_pose("shoulder_abduction", "LEFT", v))
              for v in [0.0] * 60 + list(np.linspace(0, 100, 30))
              + [100.0] * 15 + list(np.linspace(100, 0, 30)) + [0.0] * 45]
    analyzer, _ = run(frames)
    assert analyzer.reps == 0
    assert analyzer.off_plane_reps == 1


# ---------------------------------------------------------
# Compensation detector
# ---------------------------------------------------------

def detector():
    return GeometricCompensationDetector("LEFT", "shoulder_flexion")


def calibrated_detector(**trunk):
    det = detector()
    for _ in range(20):
        det.update(flex(SEATED_REST, **trunk), False, arm_at_rest=True)
    assert det.baseline_established()
    return det


def upside_down_trunk(result):
    """A MediaPipe world-landmark failure seen live: hips estimated
    above the shoulders."""
    world = result.pose_world_landmarks.landmark
    for hip in (23, 24):
        world[hip].y = -1.0
    return result


def test_implausible_pose_estimate_is_unable_not_detected():
    det = calibrated_detector()
    for _ in range(10):
        out = det.update(upside_down_trunk(flex(70)), True)
    for name in (TRUNK_IMAGE, TRUNK_DEPTH, SHOULDER_HIKING):
        assert out[name].state == UNABLE
        assert out[name].reason.startswith("implausible pose estimate")
    summary = det.summary()
    assert all(p["frames"][DETECTED] == 0
               for p in summary["patterns"].values())


def test_deviation_beyond_plausible_limit_is_unable():
    det = calibrated_detector()
    for _ in range(10):
        out = det.update(flex(70, trunk_lateral_lean=50.0), True)
    assert out[TRUNK_IMAGE].state == UNABLE
    assert out[TRUNK_IMAGE].reason.startswith("implausible measurement")


def test_real_lean_during_movement_is_detected():
    det = calibrated_detector()
    for _ in range(10):
        out = det.update(flex(70, trunk_lateral_lean=20.0), True)
    assert out[TRUNK_IMAGE].state == DETECTED
    assert out[TRUNK_IMAGE].deviation == pytest.approx(20.0, abs=1.0)


def test_baseline_follows_the_patient_repositioning_at_rest():
    """The patient settles in the chair 12° tilted between attempts;
    the next attempt, made in that new posture without leaning further,
    is not compensation."""
    det = calibrated_detector()
    for _ in range(10):
        det.update(flex(70), True)
    for _ in range(20):
        det.update(flex(SEATED_REST, trunk_lateral_lean=12.0), False,
                   arm_at_rest=True)
    for _ in range(10):
        out = det.update(flex(70, trunk_lateral_lean=12.0), True)
    assert out[TRUNK_IMAGE].state == NOT_DETECTED
    assert det.summary()["patterns"][TRUNK_IMAGE]["baseline_updates"] >= 2


def test_baseline_is_not_taken_while_the_arm_is_not_measured():
    det = detector()
    for _ in range(30):
        det.update(flex(SEATED_REST), False, arm_at_rest=False)
    assert not det.baseline_established()
    out = det.update(flex(70, trunk_lateral_lean=20.0), True)
    assert out[TRUNK_IMAGE].state == UNABLE
    assert out[TRUNK_IMAGE].reason == "no stable resting baseline"
    summary = det.summary()["patterns"][TRUNK_IMAGE]
    assert summary["assessed"] is False
    assert summary["detected_fraction"] is None     # never reported as 0


def test_baseline_samples_restart_when_an_attempt_starts():
    det = detector()
    for _ in range(10):
        det.update(flex(SEATED_REST), False, arm_at_rest=True)
    det.update(flex(70), True)
    for _ in range(10):
        det.update(flex(SEATED_REST), False, arm_at_rest=True)
    assert not det.baseline_established()      # 10 + 10, not 15 in a row
    for _ in range(5):
        det.update(flex(SEATED_REST), False, arm_at_rest=True)
    assert det.baseline_established()


def test_diagnostics_expose_every_decision_input():
    det = calibrated_detector()
    det.update(flex(70, trunk_lateral_lean=20.0), True)
    diagnostics = json.loads(json.dumps(det.diagnostics()))
    lean = diagnostics["patterns"][TRUNK_IMAGE]
    for key in ("state", "value", "baseline", "deviation", "threshold",
                "reason", "baseline_samples", "min_visibility"):
        assert key in lean
    assert diagnostics["baseline_established"] is True
    assert diagnostics["trunk_inclination"] is not None


# ---------------------------------------------------------
# Interaction: counting and compensation on the same frames
# ---------------------------------------------------------

def leaning_session(peaks, fps=30.0, movement="shoulder_flexion"):
    """Seated rest, then raises during which the trunk leans 20°."""
    frames = [flex(SEATED_REST)] * int(2 * fps)
    for peak in peaks:
        values = list(np.linspace(SEATED_REST, peak, int(fps))) \
            + [peak] * int(fps / 2) \
            + list(np.linspace(peak, SEATED_REST, int(fps)))
        frames += [make_results(body_pose(
            movement, "LEFT", v,
            trunk_lateral_lean=20.0 if v > 40 else 0.0)) for v in values]
        frames += [flex(SEATED_REST)] * int(1.5 * fps)
    return frames


def test_counting_and_compensation_work_together():
    det = detector()
    analyzer, _ = run(leaning_session([90] * 3), detector=det)
    assert analyzer.reps == 3
    lean = det.summary()["patterns"][TRUNK_IMAGE]
    assert lean["episodes"] == 3
    assert lean["baseline_updates"] >= 3       # refreshed between attempts


def test_compensation_works_when_every_repetition_is_rejected():
    """Plane validation rejects sideways raises; posture is still
    assessed because compensation uses only the attempt signal."""
    det = detector()
    analyzer, _ = run(leaning_session([100] * 3,
                                      movement="shoulder_abduction"),
                      detector=det)
    assert analyzer.reps == 0
    assert analyzer.off_plane_reps == 3
    assert det.summary()["patterns"][TRUNK_IMAGE]["episodes"] == 3


def run_widget(frames, monkeypatch, overlay=False, broken_analyzer=False):
    import src.ui.live_pose_widget as live
    from src.simulation.synthetic_pose import FakeCamera, ScriptedPose

    if overlay:
        monkeypatch.setenv(live.DEBUG_OVERLAY_ENV, "1")
    widget = live.LivePoseWidget(FLEXION, "LEFT", parameters=TARGET_60,
                                 threaded=False)
    widget.camera = FakeCamera(len(frames), width=240, height=240, fps=30)
    widget.pose = ScriptedPose(frames)
    if broken_analyzer:
        def broken(*_args, **_kwargs):
            raise RuntimeError("analyzer failure")
        monkeypatch.setattr(widget.analyzer, "update", broken)
    texts = []
    for _ in frames:
        widget.update_frame()
        texts.append((widget.compensation_label.text(),
                      widget.debug_label.text()))
    stats = widget.get_stats()
    widget.stop()
    return stats, texts


def test_live_widget_counts_and_shows_the_posture_warning(qapp,
                                                          monkeypatch):
    stats, texts = run_widget(leaning_session([90] * 3), monkeypatch,
                              overlay=True)
    assert stats["reps"] == 3
    assert any("trunk lean" in label.lower() for label, _ in texts)
    assert stats["compensation"]["patterns"][TRUNK_IMAGE]["episodes"] == 3
    debug = " ".join(d for _, d in texts)
    for fragment in ("posture: baseline ready", "lean side detected",
                     "rest 22°", "last end: resting still"):
        assert fragment in debug, fragment


def test_frame_error_never_leaves_a_stale_posture_verdict(qapp,
                                                          monkeypatch):
    stats, texts = run_widget([flex(SEATED_REST)] * 3, monkeypatch,
                              broken_analyzer=True)
    assert texts[-1][0].endswith("unable to assess (frame processing "
                                 "error)")


def test_flexion_session_persists_counts_and_compensation(qapp, temp_db,
                                                          scripted_camera):
    import src.database.db as db
    from src.ui.pages.rehab_page import RehabPage
    from tests.helpers import prescribe, seed_patient

    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Shoulder Flexion",
              "shoulder_flexion", "LEFT", {"target_reps": 3,
                                           "target_range": 60})
    page = RehabPage()
    frames = leaning_session([90] * 3)
    scripted_camera["results"] = frames
    page.start_exercise()
    for _ in frames:
        page.live_widget.update_frame()
    page.stop_exercise()

    [row] = db.get_sessions(patient_id)
    assert row["reps"] == 3
    assert row["metrics_version"] == 3       # the exercise's own version
    stored = json.loads(row["compensation_json"])
    assert stored["patterns"][TRUNK_IMAGE]["episodes"] == 3
    assert stored["patterns"][TRUNK_IMAGE]["baseline_updates"] >= 3
    assert stored["used_for_adaptation"] is False
    assert stored["validated"] is False


# ---------------------------------------------------------
# Audit fixes (D1-D3), dual baseline, metrics versioning
# ---------------------------------------------------------

def test_stillness_is_not_credited_across_tracking_loss():
    """D1: the first frame after a dropout must not end the attempt as
    'resting still'; only observed stillness counts."""
    fps = 30
    frames = (hold(SEATED_REST, 2, fps) + ramp(SEATED_REST, 90, 1, fps)
              + ramp(90, 23, 1, fps) + lost(2.0, fps) + hold(23, 2.0, fps))
    analyzer, analyses = run(frames, fps)
    back = len(frames) - int(2.0 * fps)
    ended = next(i for i in range(back, len(frames))
                 if not analyses[i].movement_active)
    assert (ended - back) / fps >= 0.95       # about 1 s of observed rest
    assert analyzer.segment_end_reason.startswith("resting still")


def implausible_variants():
    """Implausible frames with many different measured values."""
    frames = []
    for y in np.linspace(-1.2, -0.6, 7):          # trunk 100-180° off
        result = flex(70)
        for hip in (23, 24):
            result.pose_world_landmarks.landmark[hip].y = float(y)
        frames.append(result)
    frames += [flex(70, trunk_lateral_lean=v) for v in (48, 51, 55, 58)]
    return frames


def test_unable_reasons_are_stable_categories():
    """D2: one reason key per cause, whatever the measured value; the
    values themselves stay in the per-frame diagnostics."""
    det = calibrated_detector()
    inclinations = set()
    for result in implausible_variants():
        det.update(result, True)
        inclinations.add(det.diagnostics()["trunk_inclination"])
    reasons = det.summary()["patterns"][TRUNK_IMAGE]["unable_reasons"]
    assert set(reasons) == {
        "implausible pose estimate (trunk beyond 60° from vertical)",
        "implausible measurement (deviation beyond plausible limit)",
    }
    assert len(inclinations) > 2                  # values still recorded


def test_camera_failure_clears_the_posture_verdict(qapp, monkeypatch):
    """D3: an alert shown before the camera stopped must not stay."""
    import src.ui.live_pose_widget as live
    from src.simulation.synthetic_pose import FakeCamera, ScriptedPose

    frames = leaning_session([90])
    widget = live.LivePoseWidget(FLEXION, "LEFT", parameters=TARGET_60,
                                 threaded=False)
    widget.camera = FakeCamera(len(frames), width=240, height=240, fps=30)
    widget.pose = ScriptedPose(frames)
    alerts = []
    for _ in frames:
        widget.update_frame()
        alerts.append("trunk lean" in widget.compensation_label.text())
    assert any(alerts)
    for _ in range(live.MAX_CONSECUTIVE_READ_FAILURES):
        widget.update_frame()                     # camera delivers nothing
    assert widget.compensation_label.text() == live.CAMERA_LOST_FEEDBACK[0]
    widget.stop()


def drift_session(det, steps=4, step=4.0):
    """Rest posture shifts `step` degrees further sideways before each
    attempt; each attempt adds no lean beyond that rest posture."""
    for k in range(1, steps + 1):
        for _ in range(20):
            det.update(flex(SEATED_REST, trunk_lateral_lean=step * k),
                       False, arm_at_rest=True)
        for _ in range(10):
            out = det.update(flex(70, trunk_lateral_lean=step * k), True)
    return out


def test_gradual_rest_drift_stays_in_the_session_record():
    det = calibrated_detector()
    out = drift_session(det)
    lean = det.summary()["patterns"][TRUNK_IMAGE]
    # Live verdict: relative to the current rest, nothing new.
    assert out[TRUNK_IMAGE].state == NOT_DETECTED
    assert lean["episodes"] == 0
    # Session record: the 16° change since the session started remains.
    assert lean["session_start_baseline"] == pytest.approx(0.0, abs=0.5)
    assert lean["max_abs_rest_drift"] == pytest.approx(16.0, abs=0.5)
    assert lean["session_start"]["max_abs_deviation"] == pytest.approx(
        16.0, abs=0.5)
    assert lean["session_start"]["beyond_threshold_fraction"] > 0.4


def test_genuine_repositioning_refreshes_only_the_current_baseline():
    det = calibrated_detector()
    for _ in range(20):
        det.update(flex(SEATED_REST, trunk_lateral_lean=12.0), False,
                   arm_at_rest=True)
    for _ in range(10):
        out = det.update(flex(70, trunk_lateral_lean=12.0), True)
    assert out[TRUNK_IMAGE].state == NOT_DETECTED
    assert out[TRUNK_IMAGE].baseline == pytest.approx(12.0, abs=0.5)
    assert out[TRUNK_IMAGE].session_baseline == pytest.approx(0.0, abs=0.5)
    assert out[TRUNK_IMAGE].session_deviation == pytest.approx(12.0,
                                                               abs=0.5)


def test_session_start_baseline_is_never_replaced():
    det = calibrated_detector()
    first = det.patterns[TRUNK_IMAGE].session_baseline
    drift_session(det, steps=6)
    state = det.patterns[TRUNK_IMAGE]
    assert state.baseline_updates > 6
    assert state.session_baseline == first


def test_both_references_are_exposed():
    det = calibrated_detector()
    drift_session(det, steps=1)
    det.update(flex(70, trunk_lateral_lean=20.0), True)
    lean = det.diagnostics()["patterns"][TRUNK_IMAGE]
    for key in ("baseline", "current_deviation", "session_baseline",
                "session_deviation"):
        assert lean[key] is not None, key
    summary = det.summary()["patterns"][TRUNK_IMAGE]
    for key in ("baseline", "session_start_baseline", "rest_drift",
                "max_abs_rest_drift", "session_start"):
        assert key in summary, key


def test_compensation_is_identical_whatever_the_repetition_decision():
    """The same frames give the same compensation summary whether plane
    validation accepts or rejects every repetition."""
    frames = leaning_session([100] * 2, movement="shoulder_abduction")
    summaries = []
    for config in (FLEXION, replace(FLEXION, plane_check=None)):
        det = detector()
        analyzer, _ = run(frames, config=config, detector=det)
        summaries.append((analyzer.reps, det.summary()))
    (rejected_reps, rejected), (accepted_reps, accepted) = summaries
    assert rejected_reps == 0 and accepted_reps == 2
    assert rejected == accepted


# Saved sessions from before the dual reference and the categorical
# reasons (fields as stored on 2026-10-31).
HISTORICAL_COMPENSATION = {
    "method": "geometric_rules_v1", "validated": False,
    "baseline_established": True,
    "patterns": {
        TRUNK_IMAGE: {
            "label": "Trunk lean (sideways in a front view)",
            "unit": "degrees", "enter_threshold": 10.0,
            "assessed": True, "episodes": 2, "detected_fraction": 0.39,
            "max_abs_deviation": 12.67, "baseline": 0.39,
            "unable_reasons": {"shoulders/hips not visible": 131},
        },
    },
}


def test_historical_compensation_summaries_still_display():
    from src.progress.history import compensation_text

    from src.progress.history import REFERENCE_NOTE_SINGLE, REFERENCES_NOTE

    old = compensation_text(
        {"compensation_json": json.dumps(HISTORICAL_COMPENSATION)})
    # One reference only (never refreshed then): not called "current".
    assert "DETECTED vs resting posture in 39%" in old
    assert "session-start" not in old
    assert "current resting posture" not in old
    assert "held through" not in old
    assert old.endswith(REFERENCE_NOTE_SINGLE)

    det = calibrated_detector()
    drift_session(det)
    det.update(flex(70, trunk_lateral_lean=30.0), True)
    new = compensation_text(
        {"compensation_json": json.dumps(det.summary())})
    assert "vs session-start resting posture" in new
    assert "drifted up to 16" in new
    assert new.endswith(REFERENCES_NOTE)


def flexion_row(sid, version, exercise_id="shoulder_flexion"):
    from tests.test_adaptation_policy import session
    row = session(sid, version=version)
    row["exercise_id"] = exercise_id
    return row


def test_flexion_sessions_of_the_old_definition_are_not_compared():
    from src.adaptation.policy import INSUFFICIENT_DATA, evaluate
    from tests.test_adaptation_policy import PARAMS

    assert FLEXION.metrics_version == 3
    old = [flexion_row(i, 2) for i in (1, 2, 3)]
    result = evaluate(FLEXION, PARAMS, old + [flexion_row(4, 3)])
    assert result.decision == INSUFFICIENT_DATA
    reasons = {e["session_id"]: e["reason"]
               for e in result.inputs["sessions_excluded"]}
    assert set(reasons) == {1, 2, 3}
    assert all("version 2; this exercise now uses version 3" in r
               for r in reasons.values())

    result = evaluate(FLEXION, PARAMS, old + [flexion_row(4, 3),
                                              flexion_row(5, 3)])
    assert result.decision != INSUFFICIENT_DATA
    assert [s["session_id"] for s in result.inputs["sessions_used"]] == [4, 5]


def test_other_exercises_keep_version_2_and_their_history():
    from src.adaptation.policy import PolicyConfig, session_quality_issue

    for config in implemented_exercises():
        if config.exercise_id != "shoulder_flexion":
            assert config.metrics_version == 2, config.exercise_id
    row = flexion_row(1, 2, exercise_id="shoulder_abduction")
    assert session_quality_issue(row, PolicyConfig()) is None


def test_progress_history_charts_versions_separately():
    from src.progress.history import group_key

    v2 = {"metrics_version": 2, "exercise_id": "shoulder_flexion",
          "affected_side": "LEFT"}
    v3 = dict(v2, metrics_version=3)
    assert group_key(v2) != group_key(v3)
    assert group_key(v2).label() == "Shoulder Flexion (Left)"
    assert group_key(v3).label() == "Shoulder Flexion (Left) — measurement v3"


# ---------------------------------------------------------
# Final review: held frames, reference labels, version explanation
# ---------------------------------------------------------

def episode_then(frames_after):
    """Calibrated detector inside a detected lean episode, then the given
    frames (all during the attempt)."""
    det = calibrated_detector()
    for _ in range(10):
        det.update(flex(70, trunk_lateral_lean=20.0), True)
    outs = [det.update(result, True) for result in frames_after]
    return det, outs


def test_brief_visibility_loss_holds_the_warning_but_is_not_counted():
    det, outs = episode_then([empty_results()] * 3)
    lean = det.summary()["patterns"][TRUNK_IMAGE]
    assert all(o[TRUNK_IMAGE].state == DETECTED for o in outs)   # shown
    assert lean["frames_held_through_dropout"] == 3
    assert lean["frames"][DETECTED] == 6           # 10 frames - 4 onset
    assert lean["detected_fraction"] == pytest.approx(6 / 10)
    assert det.diagnostics()["patterns"][TRUNK_IMAGE][
        "held_through_dropout"] is True


def test_implausible_frame_ends_the_warning_immediately():
    det, outs = episode_then([upside_down_trunk(flex(70))])
    lean = det.summary()["patterns"][TRUNK_IMAGE]
    assert outs[0][TRUNK_IMAGE].state == UNABLE
    assert outs[0][TRUNK_IMAGE].reason.startswith("implausible pose")
    assert lean["frames_held_through_dropout"] == 0
    assert lean["frames"][DETECTED] == 6           # not incremented
    # The episode ended: a valid lean must build up again before alerting.
    out = det.update(flex(70, trunk_lateral_lean=20.0), True)
    assert out[TRUNK_IMAGE].state == NOT_DETECTED


def test_implausible_deviation_also_ends_the_warning():
    det, outs = episode_then([flex(70, trunk_lateral_lean=55.0)])
    assert outs[0][TRUNK_IMAGE].state == UNABLE
    assert det.summary()["patterns"][TRUNK_IMAGE][
        "frames_held_through_dropout"] == 0


def test_hold_is_limited_to_three_frames():
    det, outs = episode_then([empty_results()] * 5)
    assert [o[TRUNK_IMAGE].state for o in outs] == [DETECTED] * 3 + \
        [UNABLE] * 2
    lean = det.summary()["patterns"][TRUNK_IMAGE]
    assert lean["frames_held_through_dropout"] == 3
    assert lean["frames"][UNABLE] == 2


def test_frames_layout_is_unchanged_for_older_readers():
    det, _ = episode_then([empty_results()] * 2)
    lean = det.summary()["patterns"][TRUNK_IMAGE]
    assert set(lean["frames"]) == {DETECTED, NOT_DETECTED, UNABLE}


def test_held_frames_are_persisted_and_displayed(qapp, temp_db,
                                                 scripted_camera):
    import src.database.db as db
    from src.progress.history import compensation_text
    from src.ui.pages.rehab_page import RehabPage
    from tests.helpers import prescribe, seed_patient

    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Shoulder Flexion",
              "shoulder_flexion", "LEFT", {"target_reps": 3,
                                           "target_range": 60})
    frames = leaning_session([90])
    # Two frames without a pose in the middle of the leaning raise.
    middle = len(frames) - int(1.5 * 30) - 40
    frames[middle:middle + 2] = [empty_results()] * 2
    page = RehabPage()
    scripted_camera["results"] = frames
    page.start_exercise()
    for _ in frames:
        page.live_widget.update_frame()
    page.stop_exercise()

    [row] = db.get_sessions(patient_id)
    lean = json.loads(row["compensation_json"])["patterns"][TRUNK_IMAGE]
    assert lean["frames_held_through_dropout"] == 2
    text = compensation_text(row)
    assert "2 frame(s) held through brief tracking loss, not counted" in text
    assert "held through" in page.session_status.text()


def test_both_reference_labels_in_session_summary_and_history(
        qapp, temp_db, scripted_camera):
    import src.database.db as db
    from src.progress.history import REFERENCES_NOTE, compensation_text
    from src.ui.pages.rehab_page import RehabPage
    from tests.helpers import prescribe, seed_patient

    patient_id, assessment_id = seed_patient()
    prescribe(patient_id, assessment_id, "Shoulder Flexion",
              "shoulder_flexion", "LEFT", {"target_reps": 3,
                                           "target_range": 60})
    frames = leaning_session([90] * 2)
    page = RehabPage()
    scripted_camera["results"] = frames
    page.start_exercise()
    for _ in frames:
        page.live_widget.update_frame()
    page.stop_exercise()

    [row] = db.get_sessions(patient_id)
    for text in (page.session_status.text(), compensation_text(row)):
        assert "DETECTED vs current resting posture" in text
        assert "vs session-start resting posture" in text
        assert REFERENCES_NOTE in text


def version_case(comparable):
    """Three version-2 flexion sessions, then `comparable` version-3
    sessions at the same settings."""
    from src.adaptation.policy import evaluate
    from tests.test_adaptation_policy import PARAMS
    rows = [flexion_row(i, 2) for i in (1, 2, 3)]
    rows += [flexion_row(4 + i, 3) for i in range(comparable)]
    return evaluate(FLEXION, PARAMS, rows)


@pytest.mark.parametrize("comparable, needed", [(0, 2), (1, 1), (2, 0),
                                                (3, 0)])
def test_version_explanation_counts_the_sessions_still_needed(comparable,
                                                              needed):
    from src.adaptation.policy import INSUFFICIENT_DATA
    result = version_case(comparable)
    note = result.inputs["version_note"]
    assert note.startswith("Shoulder Flexion is now measured with an "
                           "updated method (version 3). Earlier sessions "
                           "remain in your history but are not compared")
    assert note in result.rationale_text()
    if needed:
        assert f"{needed} more session(s) at these settings" in note
        assert result.decision == INSUFFICIENT_DATA
    else:
        assert "more session(s)" not in note
        assert result.decision != INSUFFICIENT_DATA
    # Never any version-2 session in the calculation.
    assert all(s["session_id"] >= 4 for s in result.inputs["sessions_used"])


def test_no_version_note_without_older_sessions():
    from src.adaptation.policy import evaluate
    from tests.test_adaptation_policy import PARAMS
    result = evaluate(FLEXION, PARAMS, [flexion_row(1, 3)])
    assert result.inputs["version_note"] is None
    assert "updated method" not in result.rationale_text()


def test_patient_feedback_explains_the_version_change():
    from src.adaptation.controller import decide
    from tests.test_adaptive_controller import PARAMS, attempt
    old = []
    for i in (1, 2, 3):
        row = attempt(i)
        row["metrics_version"] = 2
        old.append(row)
    new = attempt(4)
    out = decide(FLEXION, PARAMS, new, old + [new])
    assert "updated method (version 3)" in out.feedback
    assert "1 more session(s) at these settings" in out.feedback
    assert "updated method (version 3)" in out.reason


def test_version_mismatch_feedback_gives_no_camera_advice():
    from src.adaptation.controller import UNABLE_TO_ASSESS, decide
    from tests.test_adaptive_controller import PARAMS, attempt
    old = attempt(1)
    old["metrics_version"] = 2
    out = decide(FLEXION, PARAMS, old, [old])
    assert out.action == UNABLE_TO_ASSESS
    assert "earlier measurement method (version 2)" in out.feedback
    assert "well lit" not in out.feedback and "in view" not in out.feedback


def test_camera_quality_failure_keeps_its_advice():
    from src.adaptation.controller import UNABLE_TO_ASSESS, decide
    from tests.test_adaptive_controller import PARAMS, attempt
    poor = attempt(1, measured=100)            # arm measured in 33%
    out = decide(FLEXION, PARAMS, poor, [poor])
    assert out.action == UNABLE_TO_ASSESS
    assert "well lit" in out.feedback
