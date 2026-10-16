"""
Per-attempt adaptive controller (decision support).

An attempt is one saved session of the prescribed exercise. After each
attempt the controller decides what the patient does next:

    UNABLE_TO_ASSESS     the attempt cannot be assessed reliably; nothing
                         is scored, settings are unchanged
    REPEAT               assessed, targets not met; repeat at the same
                         settings
    MAINTAIN             assessed, targets met; continue at the same
                         settings (progression needs consistent sessions)
    SIMPLIFY             easier settings are PROPOSED to the therapist
    PROPOSE_PROGRESSION  harder settings are PROPOSED to the therapist

Pipeline:
    1. attempt gate     - stale or mismatched attempt, tracking quality,
                          invalid values (policy.session_quality_issue)
    2. attempt check    - repetitions, range of motion (ROM) and movement
                          quality (DTW similarity, movement plane) of
                          THIS attempt against the active targets
    3. session policy   - policy.evaluate (conservative_v1) over the
                          recent valid sessions decides any change
    4. combination      - progression only if the policy proposes it AND
                          this attempt met every target
    5. safety guard     - bounds, one parameter by one step, direction
                          consistent with the action

Inputs use the existing project types: the ExerciseConfig, the active
prescription's parameters and session dicts as returned by
db.get_sessions (one per attempt). Thresholds are taken from
PolicyConfig; the controller adds none of its own.

Compensation: no compensation classifier is validated (Step A was
Kinect-only; Step B has not run). Any compensation output is recorded
as context and NEVER used. A missing output means "not assessed",
never "no compensation".

These rules are prototype safeguards with engineering defaults. They
are not clinically validated; every change of settings requires
therapist approval.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.adaptation.policy import (
    DECREASE,
    INCREASE,
    INSUFFICIENT_DATA,
    MAINTAIN as POLICY_MAINTAIN,
    PolicyConfig,
    evaluate,
    session_quality_issue,
)

logger = logging.getLogger(__name__)

CONTROLLER_VERSION = "attempt_controller_v1"

UNABLE_TO_ASSESS = "UNABLE_TO_ASSESS"
REPEAT = "REPEAT"
MAINTAIN = "MAINTAIN"
SIMPLIFY = "SIMPLIFY"
PROPOSE_PROGRESSION = "PROPOSE_PROGRESSION"

# Stored in adaptation_decisions.decision when no policy decision was
# made because the attempt could not be assessed.
NOT_ASSESSABLE = "NOT_ASSESSABLE"

COMPENSATION_NOT_ASSESSED = "NOT_ASSESSED"
COMPENSATION_PRESENT_NOT_USED = "PRESENT_NOT_USED"

COMPENSATION_SESSION_COLUMNS = (
    "compensation_method",
    "comp_windows_total",
    "comp_no_compensation",
    "comp_shoulder_elevation",
    "comp_trunk_rotation",
    "comp_lean_forward",
)


@dataclass(frozen=True)
class ControllerConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass
class AttemptCheck:
    """Performance of one attempt against the active targets."""

    session_id: Optional[int]
    reps: int
    target_reps: int
    completion: float
    peak: Optional[float]
    target_range: float
    rom_met: bool
    similarity: Optional[float]
    similarity_met: bool
    off_plane_reps: int
    off_plane_fraction: Optional[float]
    plane_met: bool
    measured_fraction: float
    mean_fps: float

    @property
    def reps_met(self):
        return self.completion >= 1.0

    @property
    def acceptable(self):
        return (self.reps_met and self.rom_met and self.similarity_met
                and self.plane_met)


@dataclass
class ControllerOutput:
    action: str
    feedback: str
    reason: str
    decision: str
    previous_parameters: dict
    next_attempt_parameters: dict
    proposed_parameters: dict
    changed_parameter: Optional[str] = None
    attempt: Optional[dict] = None
    compensation: dict = field(default_factory=dict)
    policy_inputs: dict = field(default_factory=dict)
    controller_version: str = CONTROLLER_VERSION

    @property
    def requires_approval(self):
        return self.proposed_parameters != self.previous_parameters

    def to_dict(self):
        data = asdict(self)
        data["requires_approval"] = self.requires_approval
        return data


# ---------------------------------------------------------
# Compensation (context only)
# ---------------------------------------------------------

def compensation_from_session(session):
    """Compensation columns of a session row (all NULL today)."""
    values = {k: session.get(k) for k in COMPENSATION_SESSION_COLUMNS}
    return values if any(v is not None for v in values.values()) else None


def describe_compensation(compensation):
    if compensation is None or compensation == {}:
        return {
            "status": COMPENSATION_NOT_ASSESSED,
            "used_in_decision": False,
            "note": "No validated compensation assessment is available. "
                    "A missing output is not evidence that no "
                    "compensation occurred.",
        }
    return {
        "status": COMPENSATION_PRESENT_NOT_USED,
        "used_in_decision": False,
        "note": "A compensation output was supplied but is not used: no "
                "compensation classifier is validated for this "
                "application.",
    }


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _parameters_of(session):
    try:
        return json.loads(session.get("parameters_json") or "null")
    except (TypeError, ValueError):
        return None


def _order_key(session):
    return (session.get("session_date") or "", session.get("session_id") or 0)


def _value(value, unit):
    return f"{value:.0f}{unit}" if unit == "°" else f"{value:.0f} {unit}"


def check_attempt(exercise, current, attempt, policy):
    target_reps = int(current["target_reps"])
    target_range = float(current["target_range"])
    reps = int(attempt.get("reps") or 0)
    off_plane = int(attempt.get("off_plane_reps") or 0)
    peak = attempt.get("max_angle")
    similarity = attempt.get("trajectory_similarity")
    attempts = reps + off_plane
    off_plane_fraction = off_plane / attempts if attempts else None

    return AttemptCheck(
        session_id=attempt.get("session_id"),
        reps=reps,
        target_reps=target_reps,
        completion=reps / target_reps,
        peak=None if peak is None else float(peak),
        target_range=target_range,
        rom_met=peak is not None and peak >= target_range,
        similarity=None if similarity is None else float(similarity),
        # No DTW score (no complete movement) is never "good enough".
        similarity_met=(
            similarity is not None
            and similarity >= policy.increase_min_similarity
        ),
        off_plane_reps=off_plane,
        off_plane_fraction=off_plane_fraction,
        plane_met=(
            off_plane_fraction is None
            or off_plane_fraction <= policy.max_off_plane_fraction
        ),
        measured_fraction=attempt["frames_measured"] / attempt["frames_total"],
        mean_fps=float(attempt["mean_fps"]),
    )


# ---------------------------------------------------------
# Feedback text (patient-facing; no compensation statements)
# ---------------------------------------------------------

def _performance_text(exercise, check):
    unit = exercise.parameter("target_range").unit
    label = exercise.angle_label.lower()
    parts = [f"You completed {check.reps} of {check.target_reps} "
             "repetitions."]
    if check.peak is None:
        parts.append(f"Your {label} could not be measured.")
    else:
        parts.append(
            f"Highest {label}: {_value(check.peak, unit)} "
            f"(target {_value(check.target_range, unit)})."
        )
    if check.similarity is None:
        parts.append("Movement similarity: not available (no complete "
                     "movement was detected).")
    else:
        parts.append(
            f"Movement similarity to the reference: "
            f"{check.similarity:.0f}/100."
        )
    return " ".join(parts)


def _focus_text(check):
    focus = []
    if not check.reps_met:
        focus.append("complete all repetitions")
    if not check.rom_met:
        focus.append("reach the target range")
    if not check.similarity_met:
        focus.append("follow the reference movement closely")
    if not check.plane_met:
        focus.append("keep your arm moving in the direction shown")
    return ("Focus on: " + ", ".join(focus) + ".") if focus else ""


def _feedback(action, exercise, check, gate_reason=None):
    if action == UNABLE_TO_ASSESS:
        return (
            f"This attempt could not be assessed reliably ({gate_reason}), "
            "so it was not scored. Check that your upper body is fully in "
            "view and well lit, then try again. Your exercise settings are "
            "unchanged."
        )

    performance = _performance_text(exercise, check)

    if action == REPEAT:
        return " ".join(filter(None, [
            performance,
            "Next: repeat the same exercise at the same settings.",
            _focus_text(check),
        ]))
    if action == MAINTAIN:
        return (
            f"{performance} You met the targets for this attempt. Next: "
            "continue at the same settings. Changes are only suggested "
            "after consistent results across sessions."
        )
    if action == PROPOSE_PROGRESSION:
        return (
            f"{performance} You met your targets consistently. A harder "
            "setting has been suggested to your therapist; until it is "
            "approved, continue at the current settings."
        )
    if action == SIMPLIFY:
        return " ".join(filter(None, [
            performance,
            "The targets were hard to reach in recent sessions. Easier "
            "settings have been suggested to your therapist; until they "
            "are approved, continue at the current settings.",
            _focus_text(check),
        ]))
    raise ValueError(action)


# ---------------------------------------------------------
# Safety guard
# ---------------------------------------------------------

def safety_violation(exercise, current, proposed, action, check):
    """Why `proposed` is unsafe for `action`, or None."""

    changed = [k for k in current if proposed.get(k) != current[k]]
    if set(proposed) != set(current):
        return "proposed parameters do not match the prescription"
    if len(changed) > 1:
        return "more than one parameter would change"

    for name in changed:
        parameter = exercise.parameter(name)
        new, old = proposed[name], current[name]
        if not parameter.minimum <= new <= parameter.maximum:
            return f"{name}={new} is outside [{parameter.minimum}, " \
                   f"{parameter.maximum}]"
        if abs(new - old) > parameter.step + 1e-9:
            return f"{name} would change by more than one step"
        if new > old and not (
            action == PROPOSE_PROGRESSION and check is not None
            and check.acceptable
        ):
            return f"{name} would increase without an acceptable attempt"
        if new < old and action != SIMPLIFY:
            return f"{name} would decrease without a simplify decision"

    if action in (UNABLE_TO_ASSESS, REPEAT, MAINTAIN) and changed:
        return f"{action} must not change parameters"
    return None


# ---------------------------------------------------------
# Controller
# ---------------------------------------------------------

def decide(exercise, current_parameters, attempt, history,
           compensation=None, config=None):
    """
    Decide the next step after one attempt.

    Args:
        exercise: ExerciseConfig of the active prescription.
        current_parameters: parameters of the active prescription.
        attempt: the session dict of the attempt just performed.
        history: this patient's session dicts for the same exercise and
            side, INCLUDING the attempt (as from db.get_sessions).
        compensation: any compensation output (or None). Recorded,
            never used.
        config: ControllerConfig.

    Returns a ControllerOutput. `next_attempt_parameters` are the
    settings of the next attempt (the active prescription: changes
    only take effect after therapist approval); `proposed_parameters`
    differ from `previous_parameters` only for SIMPLIFY and
    PROPOSE_PROGRESSION.
    """

    config = config or ControllerConfig()
    policy = config.policy
    current = exercise.resolve_parameters(current_parameters)
    compensation_info = describe_compensation(compensation)

    def unable(reason, policy_inputs=None):
        return ControllerOutput(
            action=UNABLE_TO_ASSESS,
            feedback=_feedback(UNABLE_TO_ASSESS, exercise, None, reason),
            reason=f"Attempt not assessed: {reason}. Settings unchanged.",
            decision=NOT_ASSESSABLE,
            previous_parameters=dict(current),
            next_attempt_parameters=dict(current),
            proposed_parameters=dict(current),
            attempt={"session_id": attempt.get("session_id")},
            compensation=compensation_info,
            policy_inputs=policy_inputs or {},
        )

    # 1. Attempt gate: stale or mismatched inputs.
    if attempt.get("exercise_id") != exercise.exercise_id:
        return unable("the attempt is not of the prescribed exercise")

    attempt_parameters = _parameters_of(attempt)
    if attempt_parameters is None or (
        exercise.resolve_parameters(attempt_parameters) != current
    ):
        return unable("the attempt was performed with settings other than "
                      "the active prescription")

    ordered = sorted(history, key=_order_key)
    if not ordered or ordered[-1].get("session_id") != attempt.get(
        "session_id"
    ):
        return unable("the attempt is not the most recent session")

    # Tracking quality and invalid values (same gate as the policy).
    issue = session_quality_issue(attempt, policy)
    if issue:
        return unable(issue)

    # 2. This attempt's performance.
    check = check_attempt(exercise, current, attempt, policy)

    # 3. Session policy over recent valid sessions.
    result = evaluate(exercise, current, ordered, policy)

    used = [s["session_id"] for s in result.inputs["sessions_used"]]
    if result.decision != INSUFFICIENT_DATA and (
        not used or used[-1] != attempt.get("session_id")
    ):
        return unable("the attempt was not part of the policy evaluation",
                      result.inputs)

    # 4. Combination.
    proposed = dict(current)
    changed = None
    if result.decision == INCREASE and check.acceptable:
        action = PROPOSE_PROGRESSION
        proposed, changed = dict(result.proposed_parameters), \
            result.changed_parameter
        reason = result.rationale_text()
    elif result.decision == INCREASE:
        action = REPEAT
        reason = ("Progression withheld: this attempt did not meet every "
                  "target. Settings unchanged.")
    elif result.decision == DECREASE:
        action = SIMPLIFY
        proposed, changed = dict(result.proposed_parameters), \
            result.changed_parameter
        reason = result.rationale_text()
    else:
        action = MAINTAIN if check.acceptable else REPEAT
        reason = result.rationale_text()

    # 5. Safety guard: fall back to no change rather than act unsafely.
    violation = safety_violation(exercise, current, proposed, action, check)
    if violation:
        logger.error("Adaptive controller safety guard: %s", violation)
        action = MAINTAIN if check.acceptable else REPEAT
        proposed, changed = dict(current), None
        reason = (f"Safety guard blocked a change ({violation}). "
                  "Settings unchanged.")

    decision = {
        PROPOSE_PROGRESSION: INCREASE,
        SIMPLIFY: DECREASE,
    }.get(action)
    if decision is None:
        decision = (
            result.decision if result.decision in (
                INSUFFICIENT_DATA, POLICY_MAINTAIN)
            else POLICY_MAINTAIN
        )

    return ControllerOutput(
        action=action,
        feedback=_feedback(action, exercise, check),
        reason=reason,
        decision=decision,
        previous_parameters=dict(current),
        next_attempt_parameters=dict(current),
        proposed_parameters=proposed,
        changed_parameter=changed,
        attempt=asdict(check),
        compensation=compensation_info,
        policy_inputs=dict(result.inputs, policy_decision=result.decision,
                           policy_rationale=result.rationale_text()),
    )
