"""
Conservative, rule-based difficulty adaptation (decision support).

Given the current prescription and the patient's recent sessions of the
same exercise and side, decide whether the NEXT session's parameters
should be increased, decreased or maintained. The output is a PROPOSAL:
a therapist must approve it before it changes the prescription.

Design principles
  * Only sessions recorded with the current metrics version, at the
    current parameters, with adequate tracking quality are used.
  * A change requires the same signal in the last `window` valid
    sessions; anything else (too little data, mixed signals, declining
    similarity) maintains the current parameters.
  * At most one parameter changes, by one step, within the bounds
    declared in the exercise catalogue.
  * Smoothness, experimental posture measurements and compensation
    outputs are recorded as context but NOT used: none is validated.

All thresholds are engineering defaults. They are NOT clinically
validated and must be reviewed by a therapist.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

POLICY_VERSION = "conservative_v1"

INCREASE = "INCREASE"
DECREASE = "DECREASE"
MAINTAIN = "MAINTAIN"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class PolicyConfig:
    window: int = 2
    metrics_version: int = 1
    # Tracking-quality requirements for a session to be used.
    min_measured_fraction: float = 0.8
    min_fps: float = 10.0
    max_error_fraction: float = 0.05
    # Live analysis halts after this many consecutive frame errors
    # (src/ui/live_pose_widget.py MAX_CONSECUTIVE_ERRORS) and the rest of
    # the session is not measured, so such a session is incomplete even
    # when its error fraction is small.
    max_frame_errors: int = 30
    # Increase: every window session meets all of these.
    increase_completion: float = 1.0
    increase_min_similarity: float = 70.0
    max_off_plane_fraction: float = 0.2
    # Hold if similarity fell by more than this between the last two.
    max_similarity_drop: float = 15.0
    # Decrease: every window session meets at least one of these.
    decrease_completion: float = 0.6
    decrease_range_margin_steps: float = 2.0


@dataclass
class Decision:
    decision: str
    rationale: list
    previous_parameters: dict
    proposed_parameters: dict
    changed_parameter: Optional[str] = None
    inputs: dict = field(default_factory=dict)
    policy_version: str = POLICY_VERSION

    @property
    def changes_parameters(self):
        return self.proposed_parameters != self.previous_parameters

    def rationale_text(self):
        return " ".join(self.rationale)

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, default=str)


def _parameters_of(session):
    raw = session.get("parameters_json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _finite(value):
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
    )


def invalid_values(session):
    """
    Recorded values that cannot be correct, as a list of descriptions.

    NaN compares False with every threshold, so a non-finite value would
    otherwise slip through checks such as `similarity < threshold`.
    None means "not measured" and is handled by the rules themselves.
    """

    problems = []
    for key in ("reps", "off_plane_reps", "frames_total", "frames_measured",
                "frame_errors"):
        value = session.get(key)
        if value is not None and (not _finite(value) or value < 0):
            problems.append(f"{key}={value!r}")

    for key in ("max_angle", "mean_fps"):
        value = session.get(key)
        if value is not None and not _finite(value):
            problems.append(f"{key}={value!r}")

    similarity = session.get("trajectory_similarity")
    if similarity is not None and (
        not _finite(similarity) or not 0.0 <= similarity <= 100.0
    ):
        problems.append(f"trajectory_similarity={similarity!r}")

    total = session.get("frames_total")
    measured = session.get("frames_measured")
    if _finite(total) and _finite(measured) and measured > total:
        problems.append("more measured frames than processed frames")

    return problems


def session_quality_issue(session, policy):
    """Reason a session cannot be used, or None if it is usable."""

    if session.get("metrics_version") != policy.metrics_version:
        return "recorded with an older/unknown measurement version"

    invalid = invalid_values(session)
    if invalid:
        return "invalid recorded values (" + ", ".join(invalid) + ")"

    total = session.get("frames_total") or 0
    if total <= 0:
        return "no processed frames"

    measured = session.get("frames_measured") or 0
    if measured / total < policy.min_measured_fraction:
        return (
            f"arm measured in only {100 * measured / total:.0f}% of frames"
        )

    errors = session.get("frame_errors") or 0
    if errors >= policy.max_frame_errors:
        return (
            f"{errors} frame-processing errors (analysis may have halted "
            "before the session ended)"
        )
    if errors / total > policy.max_error_fraction:
        return f"{errors} frame-processing errors"

    fps = session.get("mean_fps")
    if fps is None or fps < policy.min_fps:
        return "frame rate unknown or too low"

    if not session.get("target_reps"):
        return "no repetition target recorded"

    return None


def _session_summary(session, parameters):
    target_reps = session.get("target_reps")
    reps = session.get("reps") or 0
    return {
        "session_id": session.get("session_id"),
        "session_date": session.get("session_date"),
        "reps": reps,
        "target_reps": target_reps,
        "completion": (reps / target_reps) if target_reps else None,
        "off_plane_reps": session.get("off_plane_reps") or 0,
        "peak": session.get("max_angle"),
        "target_range": parameters.get("target_range"),
        "trajectory_similarity": session.get("trajectory_similarity"),
        # Context only; not used in the decision.
        "mean_velocity_peaks": session.get("mean_velocity_peaks"),
    }


def evaluate(exercise, current_parameters, sessions, policy=None):
    """
    Args:
        exercise: ExerciseConfig of the prescribed exercise.
        current_parameters: parameters of the active prescription.
        sessions: this patient's sessions for the same exercise and
            side (dicts as returned by db.get_sessions), any order.
        policy: PolicyConfig.

    Returns a Decision.
    """

    policy = policy or PolicyConfig()
    current = exercise.resolve_parameters(current_parameters)

    reps_param = exercise.parameter("target_reps")
    range_param = exercise.parameter("target_range")

    ordered = sorted(
        sessions,
        key=lambda s: (s.get("session_date") or "", s.get("session_id") or 0),
    )

    considered = []
    excluded = []

    for session in ordered:
        parameters = _parameters_of(session)
        if parameters is None or (
            exercise.resolve_parameters(parameters) != current
        ):
            excluded.append({
                "session_id": session.get("session_id"),
                "reason": "performed with different parameters",
            })
            continue

        issue = session_quality_issue(session, policy)
        if issue:
            excluded.append({
                "session_id": session.get("session_id"), "reason": issue,
            })
            continue

        considered.append(_session_summary(session, current))

    window = considered[-policy.window:]

    inputs = {
        "policy": asdict(policy),
        "exercise_id": exercise.exercise_id,
        "sessions_used": window,
        "sessions_excluded": excluded,
        "not_used_in_decision": [
            "movement smoothness (velocity peaks): not validated",
            "experimental posture measurements: not validated",
            "compensation classifier output: no validated classifier",
        ],
    }

    def decision(kind, reasons, proposed=None, changed=None):
        return Decision(
            decision=kind,
            rationale=reasons,
            previous_parameters=dict(current),
            proposed_parameters=dict(proposed or current),
            changed_parameter=changed,
            inputs=inputs,
        )

    if len(window) < policy.window:
        return decision(INSUFFICIENT_DATA, [
            f"Only {len(window)} usable session(s) at the current "
            f"parameters; {policy.window} are required before any change. "
            "Parameters are maintained.",
        ])

    target_range = current["target_range"]
    range_margin = policy.decrease_range_margin_steps * range_param.step

    def meets_increase(s):
        problems = []
        if s["completion"] < policy.increase_completion:
            problems.append(
                f"{s['reps']}/{s['target_reps']} target repetitions"
            )
        if s["peak"] is None or s["peak"] < target_range:
            problems.append("range target not reached")
        if (
            s["trajectory_similarity"] is None
            or s["trajectory_similarity"] < policy.increase_min_similarity
        ):
            problems.append("movement similarity below threshold")
        attempts = s["reps"] + s["off_plane_reps"]
        if attempts and (
            s["off_plane_reps"] / attempts > policy.max_off_plane_fraction
        ):
            problems.append("too many repetitions outside the movement plane")
        return problems

    def meets_decrease(s):
        reasons = []
        if s["completion"] < policy.decrease_completion:
            reasons.append("low_completion")
        if s["peak"] is not None and s["peak"] < target_range - range_margin:
            reasons.append("low_range")
        return reasons

    increase_problems = [meets_increase(s) for s in window]
    decrease_reasons = [meets_decrease(s) for s in window]

    all_increase = all(not p for p in increase_problems)
    all_decrease = all(r for r in decrease_reasons)
    any_increase = any(not p for p in increase_problems)
    any_decrease = any(r for r in decrease_reasons)

    # Sessions are identified by date in the rationale (database ids are
    # not shown to users); exact ids are in inputs["sessions_used"].
    ids = "the sessions of " + ", ".join(
        str(s["session_date"] or f"id {s['session_id']}") for s in window
    )

    # Declining similarity blocks an increase.
    sims = [s["trajectory_similarity"] for s in window]
    if all_increase and len(sims) >= 2 and (
        sims[-2] - sims[-1] > policy.max_similarity_drop
    ):
        return decision(MAINTAIN, [
            f"Targets were met in {ids}, but movement similarity "
            f"fell from {sims[-2]:.1f} to {sims[-1]:.1f}. Parameters are "
            "maintained until performance is stable.",
        ])

    if all_increase:
        peaks = [s["peak"] for s in window]
        next_range = range_param.clamp(target_range + range_param.step)
        next_reps = reps_param.clamp(current["target_reps"] + reps_param.step)

        proposed = dict(current)

        if next_range > target_range and min(peaks) >= next_range:
            proposed["target_range"] = next_range
            changed = "target_range"
            detail = (
                f"peak {exercise.angle_label.lower()} in each session "
                f"({', '.join(f'{p:.0f}' for p in peaks)} {range_param.unit}) "
                f"already reached the next level, so the range target is "
                f"raised from {target_range:g} to {next_range:g} "
                f"{range_param.unit}"
            )
        elif next_reps > current["target_reps"]:
            proposed["target_reps"] = next_reps
            changed = "target_reps"
            detail = (
                f"the repetition target is raised from "
                f"{current['target_reps']} to {next_reps}"
            )
        else:
            return decision(MAINTAIN, [
                f"All targets were met in {ids}, but both "
                "parameters are at their upper bounds. Therapist review "
                "is recommended.",
            ])

        return decision(INCREASE, [
            f"In {ids}, each session met the repetition target, the range "
            f"target and the similarity threshold; {detail}. One "
            "parameter changes by one step.",
        ], proposed, changed)

    if all_decrease and not any_increase:
        proposed = dict(current)
        low_completion = all("low_completion" in r for r in decrease_reasons)

        if low_completion and current["target_reps"] > reps_param.minimum:
            proposed["target_reps"] = reps_param.clamp(
                current["target_reps"] - reps_param.step
            )
            changed = "target_reps"
            detail = (
                f"fewer than {policy.decrease_completion:.0%} of target "
                f"repetitions were completed in each; the repetition "
                f"target is lowered from {current['target_reps']} to "
                f"{proposed['target_reps']}"
            )
        elif target_range > range_param.minimum:
            proposed["target_range"] = range_param.clamp(
                target_range - range_param.step
            )
            changed = "target_range"
            detail = (
                f"performance was below target in each; the range target "
                f"is lowered from {target_range:g} to "
                f"{proposed['target_range']:g} {range_param.unit}"
            )
        else:
            return decision(MAINTAIN, [
                f"Performance in {ids} was below target, but the parameters "
                "are already at their lower bounds. Therapist review is "
                "recommended.",
            ])

        return decision(DECREASE, [
            f"In {ids}, {detail}. One parameter changes by one "
            "step.",
        ], proposed, changed)

    if any_increase and any_decrease:
        return decision(MAINTAIN, [
            f"{ids[0].upper() + ids[1:]} gave conflicting signals (one met the "
            "targets, another was well below them). Parameters are "
            "maintained.",
        ])

    unmet = sorted({p for problems in increase_problems for p in problems})
    return decision(MAINTAIN, [
        f"{ids[0].upper() + ids[1:]} did not consistently meet the targets "
        f"({'; '.join(unmet)}) but were not low enough to reduce "
        "difficulty. Parameters are maintained.",
    ])
