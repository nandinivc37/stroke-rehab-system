# src/assessment/recommendation.py
"""
Rule-based exercise recommendation from the clinical assessment
(rules_v2). The output is a SUGGESTION: the therapist reviews it, may
choose another supported exercise and edit the starting parameters
within the catalogue bounds, and must approve it before anything is
prescribed.

Rules, applied in this order (each is a plain, testable function):

  R1 Diagnosis: only SUPPORTED_DIAGNOSES are handled.
  R2 Affected area: the exercise's joint and body region match the
     recorded affected area, and it supports the affected side.
  R3 Impairment: the impairment is one the exercise addresses.
  R4 Baseline ROM (degrees, for exercises measured in degrees): an
     exercise whose LOWEST supported range target exceeds the patient's
     recorded baseline ROM is not suggested automatically (the patient
     could not reach any target the analysis can count).
  R5 Starting parameters:
       * restriction "Pain-limited movement" or "Therapist specified":
         the most conservative start (every adaptive parameter at its
         lower bound); the therapist must set anything else;
       * otherwise the catalogue defaults, with the range target never
         above the baseline ROM (rounded down to the parameter step).
  R6 Order: the first eligible exercise in catalogue priority order is
     suggested; the other eligible exercises are listed as alternatives.

These rules are engineering choices that keep the prescription within
what the patient was measured to reach. They are NOT clinically
validated and need review by a physiotherapist. There is no severity
field in the assessment, so severity is not used.
"""

import math

from src.exercises.catalogue import implemented_exercises

RULES_VERSION = "rules_v2"

# Diagnoses that can be recorded in a clinical assessment (Assessment
# page and the Patients page diagnosis editor).
DIAGNOSIS_OPTIONS = [
    "Stroke",
    "Post-stroke hemiparesis",
    "Other",
]

# Diagnoses for which an exercise can be recommended.
SUPPORTED_DIAGNOSES = ["Stroke", "Post-stroke hemiparesis"]

# Restriction options recorded on the Assessment page.
RESTRICTION_NONE = "No major restriction"
RESTRICTION_LIMITED_RANGE = "Limited range"
RESTRICTION_PAIN = "Pain-limited movement"
RESTRICTION_THERAPIST = "Therapist specified"

# Restrictions the rules cannot interpret beyond "be cautious": start
# every adaptive parameter at its lower bound.
CONSERVATIVE_RESTRICTIONS = (RESTRICTION_PAIN, RESTRICTION_THERAPIST)

START_DEFAULT = "Default start"
START_BASELINE = "Start limited by baseline ROM"
START_LOWEST = "Lowest start (restriction)"

# Only exercises that are implemented in the exercise catalogue can be
# recommended. Previously this module kept its own list (including
# wrist, hip and knee exercises with no analysis implementation), and
# approving one of those crashed the rehabilitation session page.
EXERCISE_LIBRARY = {
    config.display_name: {
        "exercise_id": config.exercise_id,
        "joint": config.joint,
        "body_region": config.body_region,
        "supported_impairments": list(config.supported_impairments),
    }
    for config in implemented_exercises()
}


def _baseline(baseline_rom):
    """Recorded baseline ROM in degrees, or None (0 means not entered)."""
    if baseline_rom is None:
        return None
    value = float(baseline_rom)
    return value if math.isfinite(value) and value > 0 else None


def _measured_in_degrees(config):
    return config.angle_unit == "°"


def exclusion_reason(config, diagnosis, affected_side, body_region, joint,
                     impairment, baseline_rom=None):
    """Why an exercise is not eligible (R1-R4), or None if it is."""

    if diagnosis not in SUPPORTED_DIAGNOSES:
        return "diagnosis not supported"
    if config.joint != joint or config.body_region != body_region:
        return f"does not train the affected {str(joint).lower()}"
    if affected_side and affected_side.upper() not in config.sides:
        return f"does not support side {affected_side}"
    if impairment not in config.supported_impairments:
        return f"does not address '{impairment}'"

    baseline = _baseline(baseline_rom)
    if baseline is not None and _measured_in_degrees(config):
        lowest = config.parameter("target_range").minimum
        if baseline < lowest:
            return (
                f"baseline ROM {baseline:g}° is below the lowest range "
                f"target the analysis supports ({lowest:g}°)"
            )
    return None


def starting_parameters(config, baseline_rom=None, restrictions=None):
    """
    (parameters, start label, explanation) for a new prescription of
    `config` (R5). Parameters are always within the catalogue bounds.
    """

    if restrictions in CONSERVATIVE_RESTRICTIONS:
        parameters = {
            p.name: p.clamp(p.minimum) for p in config.adaptive_parameters
        }
        return parameters, START_LOWEST, (
            f"restriction '{restrictions}': every target starts at its "
            "lower bound; the therapist should set the parameters"
        )

    parameters = config.default_parameters()
    baseline = _baseline(baseline_rom)
    target = config.parameter("target_range")

    if (
        baseline is not None
        and _measured_in_degrees(config)
        and baseline < parameters["target_range"]
    ):
        steps = math.floor((baseline - target.minimum) / target.step + 1e-9)
        parameters["target_range"] = target.clamp(
            target.minimum + steps * target.step
        )
        return parameters, START_BASELINE, (
            f"range target {parameters['target_range']:g}° set at or below "
            f"the baseline ROM of {baseline:g}°"
        )

    note = "catalogue default targets"
    if baseline is not None and not _measured_in_degrees(config):
        note += (
            f" (baseline ROM in degrees does not apply to "
            f"{config.angle_unit})"
        )
    return parameters, START_DEFAULT, note


def recommend_exercise(
    diagnosis,
    affected_side,
    body_region,
    joint,
    impairment,
    baseline_rom=None,
    restrictions=None,
):
    """
    Generate a candidate rehabilitation exercise.

    This is a rule-based recommendation system.
    The therapist must review and approve the recommendation
    before it is assigned to the patient.

    Returns a dict with success, exercise, exercise_id, difficulty
    (starting-point label), parameters, reason, alternatives, excluded
    (exercise -> reason) and rules_version.
    """

    excluded = {}
    eligible = []

    for config in implemented_exercises():
        reason = exclusion_reason(
            config, diagnosis, affected_side, body_region, joint,
            impairment, baseline_rom,
        )
        if reason is None:
            eligible.append(config)
        else:
            excluded[config.display_name] = reason

    base = {
        "rules_version": RULES_VERSION,
        "excluded": excluded,
        "alternatives": [],
    }

    if diagnosis not in SUPPORTED_DIAGNOSES:
        return dict(base, success=False, exercise=None, difficulty=None,
                    parameters=None,
                    reason="Diagnosis is not currently supported.")

    if not eligible:
        return dict(base, success=False, exercise=None, difficulty=None,
                    parameters=None, reason=(
                        "No suitable exercise is currently available "
                        "for the selected clinical profile."
                    ))

    config = eligible[0]
    parameters, start, note = starting_parameters(
        config, baseline_rom, restrictions
    )
    targets = ", ".join(
        f"{p.label.lower()} {parameters[p.name]:g} {p.unit}".rstrip()
        for p in config.adaptive_parameters
    )

    reason = (
        f"{config.display_name} recommended for "
        f"{str(affected_side).lower()} {str(joint).lower()} based on the "
        f"recorded diagnosis and {str(impairment).lower()}. Starting "
        f"targets: {targets} ({note})."
    )

    return dict(
        base,
        success=True,
        exercise=config.display_name,
        exercise_id=config.exercise_id,
        difficulty=start,
        parameters=parameters,
        reason=reason,
        alternatives=[c.display_name for c in eligible[1:]],
    )


def override_note(suggested_exercise, suggested_parameters,
                  approved_config, approved_parameters):
    """
    Plain-language record of what the therapist changed relative to the
    suggestion, or "" when the suggestion was approved unchanged.
    """

    changes = []
    if suggested_exercise != approved_config.display_name:
        changes.append(
            f"prescribed {approved_config.display_name} instead of the "
            f"suggested {suggested_exercise}"
        )
    for p in approved_config.adaptive_parameters:
        suggested = (suggested_parameters or {}).get(p.name)
        approved = approved_parameters[p.name]
        if (
            suggested_exercise == approved_config.display_name
            and suggested is not None and suggested != approved
        ):
            changes.append(
                f"{p.label.lower()} {suggested:g} -> {approved:g} {p.unit}"
                .rstrip()
            )
    if not changes:
        return ""
    targets = ", ".join(
        f"{p.label.lower()} {approved_parameters[p.name]:g} {p.unit}".rstrip()
        for p in approved_config.adaptive_parameters
    )
    return (
        "Therapist override: " + "; ".join(changes)
        + f". Approved targets: {targets}."
    )
