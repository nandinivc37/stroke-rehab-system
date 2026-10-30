"""
Per-attempt adaptive controller: rules, safety and uncertainty handling,
and propagation through the real session page and database.

Mock assessment inputs (session dicts) and synthetic sessions only:
this verifies the software behaviour, not clinical appropriateness.
"""

import json
import math

import pytest

import src.adaptation.controller as controller
import src.database.db as db
from src.adaptation.controller import (
    MAINTAIN,
    PROPOSE_PROGRESSION,
    REPEAT,
    SIMPLIFY,
    UNABLE_TO_ASSESS,
    ControllerConfig,
    decide,
    safety_violation,
)
from src.adaptation.policy import (
    INCREASE,
    Decision,
    PolicyConfig,
    session_quality_issue,
)
from src.adaptation.workflow import (
    decision_history,
    pending_proposals,
    review_proposal,
)
from src.exercises.catalogue import get_exercise_config
from tests.helpers import perform, prescribe, seed_patient

FLEXION = get_exercise_config("shoulder_flexion")
PARAMS = {"target_reps": 10, "target_range": 80.0}


def check_realistic(reps, off_plane, peak, params):
    """
    The repetition counter only counts once the angle reaches the range
    target (rep threshold = target_range), so a session with counted
    repetitions always has a peak at or above it. Fixtures that violate
    this test states the real pipeline cannot produce.
    """
    if (reps or off_plane) and (peak is None or peak < params["target_range"]):
        raise ValueError(
            f"unrealistic fixture: {reps} reps with peak {peak} below the "
            f"range target {params['target_range']}"
        )


def attempt(sid, reps=10, peak=95.0, similarity=85.0, params=None,
            off_plane=0, frames=300, measured=300, errors=0, fps=25.0,
            exercise_id="shoulder_flexion", **extra):
    params = params or PARAMS
    check_realistic(reps, off_plane, peak, params)
    row = {
        "session_id": sid,
        "session_date": f"2026-10-{sid:02d} 10:00:00",
        "exercise_id": exercise_id,
        "metrics_version": 2,
        "session_status": "COMPLETE",
        "parameters_json": json.dumps(params),
        "reps": reps,
        "target_reps": params["target_reps"],
        "off_plane_reps": off_plane,
        "max_angle": peak,
        "peak_rom": peak,
        "trajectory_similarity": similarity,
        "frames_total": frames,
        "frames_measured": measured,
        "frame_errors": errors,
        "mean_fps": fps,
    }
    row.update(extra)
    return row


def run(history, params=PARAMS, compensation=None, config=None):
    """Decide after the last session in `history`."""
    return decide(FLEXION, params, history[-1], history, compensation,
                  config)


def no_increase(out):
    return all(out.proposed_parameters[k] <= out.previous_parameters[k]
               for k in out.previous_parameters)


# ---------------------------------------------------------
# Good repeated performance, bounded progression
# ---------------------------------------------------------

def test_single_good_attempt_maintains():
    out = run([attempt(1)])
    assert out.action == MAINTAIN
    assert out.decision == "INSUFFICIENT_DATA"
    assert out.proposed_parameters == out.previous_parameters
    assert "met the targets" in out.feedback


def test_consistent_good_attempts_propose_one_step():
    out = run([attempt(1), attempt(2)])
    assert out.action == PROPOSE_PROGRESSION
    assert out.decision == INCREASE
    assert out.requires_approval
    # Peaks 95 >= next range level 85: range rises by one step only.
    assert out.proposed_parameters == {"target_reps": 10,
                                       "target_range": 85.0}
    # Until approval the next attempt keeps the active prescription.
    assert out.next_attempt_parameters == PARAMS
    assert "therapist" in out.feedback


def test_repeated_progression_stops_at_upper_bounds():
    params = dict(PARAMS)
    history = []
    proposals = 0
    for sid in range(1, 80):
        history.append(attempt(sid, reps=params["target_reps"], peak=150.0,
                               params=params))
        out = run(history, params)
        if out.action == PROPOSE_PROGRESSION:
            proposals += 1
            [name] = [k for k in params
                      if out.proposed_parameters[k] != params[k]]
            step = FLEXION.parameter(name).step
            assert out.proposed_parameters[name] - params[name] == step
            params = out.proposed_parameters  # therapist approves
            history = []  # sessions at old parameters no longer count
    assert params == {"target_reps": 30, "target_range": 150.0}
    assert out.action == MAINTAIN and not out.requires_approval
    assert "upper bounds" in out.reason
    assert proposals == (150 - 80) / 5 + (30 - 10) / 2


# ---------------------------------------------------------
# Poor ROM and poor movement quality
# ---------------------------------------------------------

def test_insufficient_rom_never_progresses():
    out = run([attempt(1), attempt(2), attempt(3, reps=0, peak=75.0)])
    assert out.action == REPEAT and no_increase(out)
    assert "reach the target range" in out.feedback


def test_poor_rom_in_recent_attempts_simplifies_the_range():
    # Below the range target the counter credits no repetitions.
    out = run([attempt(1, reps=0, peak=40.0), attempt(2, reps=0, peak=40.0)])
    assert out.action == SIMPLIFY
    assert out.proposed_parameters == {"target_reps": 10,
                                       "target_range": 75.0}
    assert out.next_attempt_parameters == PARAMS
    assert no_increase(out)


def test_few_repetitions_at_full_range_simplifies_the_repetitions():
    out = run([attempt(1, reps=2, peak=95.0), attempt(2, reps=3, peak=92.0)])
    assert out.action == SIMPLIFY
    assert out.proposed_parameters == {"target_reps": 8,
                                       "target_range": 80.0}
    assert no_increase(out)


@pytest.mark.parametrize("poor", [
    dict(similarity=50.0),               # DTW similarity below threshold
    dict(off_plane=5, reps=10),          # 33% of reps outside the plane
])
def test_poor_movement_quality_never_progresses(poor):
    out = run([attempt(1), attempt(2), attempt(3, **poor)])
    assert out.action == REPEAT and no_increase(out)
    assert "Focus on" in out.feedback


# ---------------------------------------------------------
# Tracking confidence, missing / invalid inputs
# ---------------------------------------------------------

@pytest.mark.parametrize("low", [
    dict(measured=120),              # arm tracked in 40% of frames
    dict(fps=5.0),
    dict(fps=None),
    dict(errors=30, frames=3000, measured=3000),  # analysis halted
    dict(frames=0, measured=0),
])
def test_low_tracking_confidence_is_not_assessed(low):
    # Two earlier good attempts would justify progression on their own.
    out = run([attempt(1), attempt(2), attempt(3, **low)])
    assert out.action == UNABLE_TO_ASSESS
    assert out.decision == "NOT_ASSESSABLE"
    assert out.proposed_parameters == out.next_attempt_parameters == PARAMS
    assert "not scored" in out.feedback
    assert "could not be assessed" in out.feedback


@pytest.mark.parametrize("invalid", [
    dict(similarity=float("nan")),
    dict(similarity=150.0),
    dict(similarity=-3.0),
    dict(peak=float("nan")),
    dict(peak=float("inf")),
    dict(reps=-1),
    dict(fps=float("nan")),
    dict(measured=400),
])
def test_invalid_assessment_values_are_not_assessed(invalid):
    out = run([attempt(1), attempt(2), attempt(3, **invalid)])
    assert out.action == UNABLE_TO_ASSESS
    assert "invalid recorded values" in out.reason
    assert not out.requires_approval


def test_invalid_earlier_session_cannot_support_progression():
    issue = session_quality_issue(attempt(1, similarity=float("nan")),
                                  PolicyConfig())
    assert "invalid" in issue
    out = run([attempt(1, similarity=float("nan")), attempt(2)])
    assert out.action == MAINTAIN and not out.requires_approval


def test_missing_dtw_blocks_progression_without_inventing_a_score():
    out = run([attempt(1), attempt(2), attempt(3, similarity=None)])
    assert out.action == REPEAT and no_increase(out)
    assert out.attempt["similarity"] is None
    assert "not available" in out.feedback


def test_missing_rom_blocks_progression():
    out = run([attempt(1), attempt(2), attempt(3, reps=0, peak=None)])
    assert out.action == REPEAT and no_increase(out)
    assert "could not be measured" in out.feedback


@pytest.mark.parametrize("stale, message", [
    (dict(params={"target_reps": 8, "target_range": 80.0}), "settings"),
    (dict(exercise_id="shoulder_abduction"), "prescribed exercise"),
    (dict(parameters_json="not json"), "settings"),
])
def test_stale_or_mismatched_attempt_is_not_assessed(stale, message):
    out = run([attempt(1), attempt(2), attempt(3, **stale)])
    assert out.action == UNABLE_TO_ASSESS and message in out.reason


def test_attempt_that_is_not_the_latest_is_not_assessed():
    history = [attempt(1), attempt(2), attempt(3)]
    out = decide(FLEXION, PARAMS, history[1], history)
    assert out.action == UNABLE_TO_ASSESS
    assert "most recent" in out.reason


# ---------------------------------------------------------
# Difficulty bounds and the safety guard
# ---------------------------------------------------------

def test_lower_bounds_are_respected():
    bottom = {"target_reps": 3, "target_range": 50.0}
    out = run([attempt(1, reps=0, peak=20.0, params=bottom),
               attempt(2, reps=0, peak=20.0, params=bottom)], bottom)
    assert out.action == REPEAT and not out.requires_approval
    assert "lower bound" in out.reason


@pytest.mark.parametrize("proposed, action, message", [
    ({"target_reps": 10, "target_range": 155.0}, PROPOSE_PROGRESSION,
     "outside"),
    ({"target_reps": 10, "target_range": 90.0}, PROPOSE_PROGRESSION,
     "more than one step"),
    ({"target_reps": 12, "target_range": 85.0}, PROPOSE_PROGRESSION,
     "more than one parameter"),
    ({"target_reps": 10, "target_range": 85.0}, MAINTAIN, "increase without"),
    ({"target_reps": 8, "target_range": 80.0}, REPEAT, "decrease"),
    ({"target_reps": 8, "target_range": 80.0}, PROPOSE_PROGRESSION,
     "decrease"),
    ({"target_reps": 12, "target_range": 80.0}, SIMPLIFY,
     "increase without"),
])
def test_safety_guard_rejects_unsafe_changes(proposed, action, message):
    check = controller.check_attempt(FLEXION, PARAMS, attempt(1),
                                     PolicyConfig())
    current = {"target_reps": 10, "target_range": 80.0}
    if proposed["target_range"] == 155.0:
        current = {"target_reps": 10, "target_range": 150.0}
    assert message in safety_violation(FLEXION, current, proposed, action,
                                       check)


def test_guard_blocks_a_faulty_policy_increase(monkeypatch):
    def faulty(exercise, current, history, policy):
        return Decision(
            decision=INCREASE, rationale=["faulty"],
            previous_parameters=dict(current),
            proposed_parameters={"target_reps": 30, "target_range": 150.0},
            inputs={"sessions_used": [{"session_id": 2}]},
        )
    monkeypatch.setattr(controller, "evaluate", faulty)
    out = run([attempt(1), attempt(2)])
    assert out.action == MAINTAIN and not out.requires_approval
    assert "Safety guard" in out.reason


# ---------------------------------------------------------
# Compensation outputs are never decision inputs
# ---------------------------------------------------------

COMPENSATION_VARIANTS = [
    None,
    {},
    {"compensation_method": "bilstm_shared_v1", "comp_windows_total": 100,
     "comp_no_compensation": 100},                    # "none detected"
    {"compensation_method": "rf_shared_v1", "comp_windows_total": 100,
     "comp_shoulder_elevation": 90},                  # "severe"
    {"label": "uncertain", "confidence": 0.31},
]


@pytest.mark.parametrize("history", [
    [attempt(1), attempt(2)],                                   # progress
    [attempt(1), attempt(2), attempt(3, reps=0, peak=70.0)],    # repeat
    [attempt(1, reps=0, peak=40.0), attempt(2, reps=0, peak=40.0)],
])
def test_compensation_output_cannot_change_the_decision(history):
    outputs = [run(history, compensation=c) for c in COMPENSATION_VARIANTS]
    reference = outputs[0].to_dict()
    for out in outputs[1:]:
        data = out.to_dict()
        assert data.pop("compensation")["used_in_decision"] is False
        assert data == {k: v for k, v in reference.items()
                        if k != "compensation"}


def test_missing_compensation_is_not_absence_of_compensation():
    out = run([attempt(1), attempt(2)])
    assert out.compensation["status"] == "NOT_ASSESSED"
    assert "not evidence" in out.compensation["note"]
    # Patient-facing feedback says nothing about compensation.
    assert "compensat" not in out.feedback.lower()


def test_no_compensation_report_cannot_rescue_a_poor_attempt():
    clean = {"compensation_method": "bilstm_shared_v1",
             "comp_windows_total": 50, "comp_no_compensation": 50}
    out = run([attempt(1), attempt(2), attempt(3, similarity=40.0)],
              compensation=clean)
    assert out.action == REPEAT and no_increase(out)
    assert out.compensation["status"] == "PRESENT_NOT_USED"


def test_config_thresholds_come_from_policy_config():
    strict = ControllerConfig(policy=PolicyConfig(increase_min_similarity=90))
    out = run([attempt(1), attempt(2)], config=strict)
    assert out.action == REPEAT and no_increase(out)


# ---------------------------------------------------------
# Real session page + database: persistence and propagation
# ---------------------------------------------------------

@pytest.fixture
def abduction(temp_db):
    patient_id, assessment_id = seed_patient()
    rec_id = prescribe(
        patient_id, assessment_id, "Shoulder Abduction",
        "shoulder_abduction", "LEFT",
        {"target_reps": 3, "target_range": 80},
    )
    return patient_id, rec_id


def test_attempts_persist_action_and_feedback(qapp, abduction,
                                              scripted_camera):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, _ = abduction
    page = RehabPage()

    first = perform(page, scripted_camera, [84, 84])        # 2 of 3 reps
    assert first["action"] == REPEAT
    assert first["feedback"] in page.session_status.text()
    assert "repeat the same exercise" in page.session_status.text()

    second = perform(page, scripted_camera, [84, 84, 84])
    assert second["action"] == MAINTAIN

    history = decision_history(patient_id)
    assert [h["action"] for h in history] == [MAINTAIN, REPEAT]
    stored = json.loads(history[0]["inputs_json"])["controller"]
    assert stored["controller_version"] == "attempt_controller_v1"
    # Sessions now carry the rule-based compensation summary: recorded,
    # never used in the decision.
    assert stored["compensation"]["status"] == "PRESENT_NOT_USED"
    assert stored["compensation"]["used_in_decision"] is False
    assert stored["feedback"] == history[0]["feedback"]
    sessions = db.get_sessions(patient_id)
    assert [h["session_id"] for h in history] == [
        s["session_id"] for s in reversed(sessions)
    ]


def test_unassessable_attempt_does_not_replace_pending_progression(
    qapp, abduction, scripted_camera
):
    from src.simulation.session_simulator import simulate_session
    from src.ui.pages.rehab_page import RehabPage

    patient_id, original = abduction
    page = RehabPage()
    perform(page, scripted_camera, [84, 84, 84])
    progression = perform(page, scripted_camera, [84, 84, 84])
    assert progression["action"] == PROPOSE_PROGRESSION
    [pending] = pending_proposals(patient_id)

    # Arm tracked in only half of the frames.
    scripted_camera["results"] = simulate_session(
        "shoulder_abduction", "LEFT", [84, 84, 84], drop_every=2)
    page.start_exercise()
    for _ in scripted_camera["results"]:
        page.live_widget.update_frame()
    page.stop_exercise()

    assert page.last_adaptation["action"] == UNABLE_TO_ASSESS
    assert page.last_adaptation["proposal_id"] is None
    assert "could not be assessed" in page.session_status.text()
    assert [p["recommendation_id"] for p in pending_proposals(patient_id)] \
        == [pending["recommendation_id"]]
    assert page.current_prescription_id == original


def test_recommendation_propagates_to_the_next_attempt(qapp, abduction,
                                                       scripted_camera):
    from src.ui.pages.rehab_page import RehabPage

    patient_id, original = abduction
    page = RehabPage()

    # Repeat / maintain: the next attempt runs at the same settings.
    out = perform(page, scripted_camera, [40, 40, 40])
    assert out["action"] == REPEAT
    assert page.current_parameters == out["next_attempt_parameters"]

    # Simplify: proposed, then applied to the next attempt once approved.
    out = perform(page, scripted_camera, [40, 40, 40])
    assert out["action"] == SIMPLIFY
    assert out["proposed_parameters"] == {"target_reps": 3,
                                          "target_range": 75.0}
    assert page.current_parameters == out["next_attempt_parameters"] == {
        "target_reps": 3, "target_range": 80.0}
    assert "awaiting therapist approval" in page.session_status.text()

    review_proposal(out["proposal_id"], True, "Therapist A")
    page.refresh()
    assert page.current_parameters == {"target_reps": 3,
                                       "target_range": 75.0}

    # The next attempt is analysed with the new target, then reassessed.
    perform(page, scripted_camera, [78, 78, 78])
    last = db.get_sessions(patient_id)[-1]
    assert last["prescription_id"] == out["proposal_id"]
    assert json.loads(last["parameters_json"])["target_range"] == 75.0
    assert last["reps"] == 3            # 78° counts against 75°, not 80°
    assert page.last_adaptation["action"] == MAINTAIN
    assert page.last_adaptation["decision"] == "INSUFFICIENT_DATA"
    assert not math.isnan(last["trajectory_similarity"])
