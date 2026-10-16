"""WS5: every branch of the conservative adaptation policy."""

import json

import pytest

from src.adaptation.policy import (
    DECREASE,
    INCREASE,
    INSUFFICIENT_DATA,
    MAINTAIN,
    PolicyConfig,
    evaluate,
)
from src.exercises.catalogue import get_exercise_config

FLEXION = get_exercise_config("shoulder_flexion")
PARAMS = {"target_reps": 10, "target_range": 80.0}


def session(sid, reps=10, peak=95.0, similarity=85.0, params=None,
            target_reps=10, off_plane=0, version=1, frames=300,
            measured=300, errors=0, fps=25.0):
    return {
        "session_id": sid,
        "session_date": f"2026-10-{sid:02d} 10:00:00",
        "metrics_version": version,
        "parameters_json": json.dumps(params or PARAMS),
        "reps": reps,
        "target_reps": target_reps,
        "off_plane_reps": off_plane,
        "max_angle": peak,
        "trajectory_similarity": similarity,
        "frames_total": frames,
        "frames_measured": measured,
        "frame_errors": errors,
        "mean_fps": fps,
        "mean_velocity_peaks": 2.0,
    }


def decide(sessions, params=PARAMS, policy=None):
    return evaluate(FLEXION, params, sessions, policy)


def test_first_time_user_is_insufficient_data():
    result = decide([])
    assert result.decision == INSUFFICIENT_DATA
    assert result.proposed_parameters == result.previous_parameters
    assert not result.changes_parameters


def test_one_session_is_insufficient():
    assert decide([session(1)]).decision == INSUFFICIENT_DATA


def test_two_good_sessions_increase_reps_by_one_step():
    result = decide([session(1, peak=82), session(2, peak=83)])
    assert result.decision == INCREASE
    assert result.changed_parameter == "target_reps"
    assert result.proposed_parameters == {"target_reps": 12,
                                          "target_range": 80.0}


def test_range_raised_only_when_already_achieved():
    result = decide([session(1, peak=90), session(2, peak=88)])
    assert result.decision == INCREASE
    assert result.changed_parameter == "target_range"
    assert result.proposed_parameters == {"target_reps": 10,
                                          "target_range": 85.0}


def test_only_one_parameter_changes():
    result = decide([session(1, peak=120), session(2, peak=120)])
    changed = [k for k in PARAMS
               if result.proposed_parameters[k] != result.previous_parameters[k]]
    assert changed == [result.changed_parameter]


def test_increase_blocked_by_low_similarity():
    result = decide([session(1), session(2, similarity=60)])
    assert result.decision == MAINTAIN
    assert "similarity" in result.rationale_text()


def test_increase_blocked_by_declining_similarity():
    result = decide([session(1, similarity=95), session(2, similarity=75)])
    assert result.decision == MAINTAIN
    assert "fell" in result.rationale_text()


def test_increase_blocked_by_off_plane_repetitions():
    result = decide([session(1), session(2, off_plane=5)])
    assert result.decision == MAINTAIN


def test_two_low_completion_sessions_decrease_reps():
    result = decide([session(1, reps=3), session(2, reps=4)])
    assert result.decision == DECREASE
    assert result.changed_parameter == "target_reps"
    assert result.proposed_parameters["target_reps"] == 8


def test_low_range_decreases_range_target():
    result = decide([session(1, reps=7, peak=60), session(2, reps=7, peak=62)])
    assert result.decision == DECREASE
    assert result.changed_parameter == "target_range"
    assert result.proposed_parameters["target_range"] == 75.0


def test_conflicting_sessions_maintain():
    result = decide([session(1), session(2, reps=2)])
    assert result.decision == MAINTAIN
    assert "conflicting" in result.rationale_text()


def test_middling_performance_maintains():
    result = decide([session(1, reps=8), session(2, reps=9)])
    assert result.decision == MAINTAIN


def test_only_latest_window_is_used():
    old_bad = [session(i, reps=1) for i in range(1, 5)]
    result = decide(old_bad + [session(5), session(6)])
    assert result.decision == INCREASE
    assert [s["session_id"] for s in result.inputs["sessions_used"]] == [5, 6]


@pytest.mark.parametrize("bad", [
    dict(version=None),
    dict(measured=100),
    dict(errors=50),
    # Possibly halted: 30 errors is only 1% of frames but incomplete.
    dict(errors=30, frames=3000, measured=3000),
    dict(fps=None),
    dict(fps=5.0),
    dict(frames=0, measured=0),
    dict(target_reps=None),
])
def test_unreliable_sessions_are_excluded(bad):
    result = decide([session(1), session(2, **bad)])
    assert result.decision == INSUFFICIENT_DATA
    assert result.inputs["sessions_excluded"][0]["session_id"] == 2


def test_sessions_at_other_parameters_are_excluded():
    other = {"target_reps": 8, "target_range": 80.0}
    result = decide([session(1, params=other), session(2)])
    assert result.decision == INSUFFICIENT_DATA
    assert "different parameters" in (
        result.inputs["sessions_excluded"][0]["reason"]
    )


def test_upper_bounds_respected():
    top = {"target_reps": 30, "target_range": 150.0}
    result = decide([session(1, reps=30, target_reps=30, peak=150, params=top),
                     session(2, reps=30, target_reps=30, peak=150, params=top)],
                    params=top)
    assert result.decision == MAINTAIN
    assert "upper bounds" in result.rationale_text()


def test_lower_bounds_respected():
    bottom = {"target_reps": 3, "target_range": 50.0}
    result = decide([session(1, reps=0, target_reps=3, peak=20, params=bottom),
                     session(2, reps=0, target_reps=3, peak=20, params=bottom)],
                    params=bottom)
    assert result.decision == MAINTAIN
    assert "lower bounds" in result.rationale_text()


def test_unvalidated_signals_are_recorded_as_not_used():
    result = decide([session(1), session(2)])
    not_used = " ".join(result.inputs["not_used_in_decision"])
    assert "compensation" in not_used and "posture" in not_used
    # Decision inputs carry no compensation or posture fields.
    for used in result.inputs["sessions_used"]:
        assert not any(
            key.startswith("comp_") or "compensation" in key
            or "posture" in key
            for key in used
        )


def test_policy_thresholds_are_configurable():
    strict = PolicyConfig(window=3)
    assert decide([session(1), session(2)], policy=strict).decision == (
        INSUFFICIENT_DATA
    )


def test_decision_is_serialisable():
    result = decide([session(1), session(2)])
    assert json.loads(result.to_json())["decision"] == INCREASE


def test_error_cap_matches_live_halt_threshold():
    import src.ui.live_pose_widget as live
    assert PolicyConfig().max_frame_errors == live.MAX_CONSECUTIVE_ERRORS
