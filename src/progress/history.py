"""
Session-history helpers for the Progress and Dashboard pages.

Rules:
  * Missing measurements are shown as "—", never as 0.
  * Sessions recorded by the original implementation (metrics_version
    NULL) are grouped separately from current measurements: their angle
    columns were produced by different code (2D image-plane arm angle,
    or the older elbow counter) and are not comparable.
"""

import json
from dataclasses import dataclass

from src.exercises.catalogue import get_exercise_config

MISSING = "—"


@dataclass(frozen=True)
class GroupKey:
    exercise: str          # exercise_id, or the stored name for legacy rows
    side: str              # "" when unknown (legacy rows)
    legacy: bool

    def label(self):
        config = get_exercise_config(self.exercise)
        name = config.display_name if config else self.exercise
        if self.legacy:
            return f"{name} — legacy measurement (before metrics v1)"
        return f"{name} ({self.side.title()})"


def group_key(session):
    if session.get("metrics_version") is None:
        return GroupKey(session.get("exercise") or "Unknown", "", True)
    return GroupKey(
        session.get("exercise_id") or session.get("exercise") or "Unknown",
        session.get("affected_side") or "",
        False,
    )


def group_sessions(sessions):
    """Ordered {GroupKey: [sessions]}; current groups before legacy."""

    groups = {}
    for session in sessions:
        groups.setdefault(group_key(session), []).append(session)

    return dict(sorted(
        groups.items(),
        key=lambda item: (item[0].legacy, item[0].label()),
    ))


def fmt(value, pattern="{:.1f}", suffix=""):
    if value is None:
        return MISSING
    return pattern.format(value) + suffix


def pose_coverage(session):
    total = session.get("frames_total")
    measured = session.get("frames_measured")
    if not total or measured is None:
        return None
    return 100.0 * measured / total


def reps_text(session):
    reps = session.get("reps")
    target = session.get("target_reps")
    if reps is None:
        return MISSING
    return f"{reps} / {target}" if target else str(reps)


def unit_for(session):
    config = get_exercise_config(session.get("exercise_id"))
    if session.get("metrics_version") is None or config is None:
        return "°"
    return config.angle_unit


def short_label(session):
    """Exercise name (and side) without the group's legacy annotation."""
    key = group_key(session)
    config = get_exercise_config(key.exercise)
    name = config.display_name if config else key.exercise
    return f"{name} ({key.side.title()})" if key.side else name


# Pseudo-field: the peak shown for a session (see peak_value).
PEAK_FIELD = "peak"


def peak_value(session):
    """
    Peak range shown for a session: peak ROM over in-plane movements for
    metrics v2 (None when no such movement was measured), the raw
    session maximum for older sessions, whose definition it was.
    """
    if (session.get("metrics_version") or 0) >= 2:
        return session.get("peak_rom")
    return session.get("max_angle")


def peak_series(sessions):
    xs, ys = [], []
    for index, session in enumerate(sessions, start=1):
        value = peak_value(session)
        if value is not None:
            xs.append(index)
            ys.append(value)
    return xs, ys


def status_text(session):
    status = session.get("session_status")
    if status is None or status == "COMPLETE":
        return "complete" if status else MISSING
    return status.replace("_", " ").lower()


def table_row(session):
    """Display strings for one session row."""

    unit = unit_for(session)
    return [
        session.get("session_date") or MISSING,
        short_label(session),
        reps_text(session),
        fmt(peak_value(session), suffix=f" {unit}".rstrip()),
        # Movement score Q (DTW shape + range completion), 0-100; the
        # column name trajectory_similarity is historical.
        fmt(session.get("trajectory_similarity")),
        fmt(session.get("dtw_similarity")),
        fmt(session.get("valid_segments"), "{:d}"),
        fmt(pose_coverage(session), "{:.0f}", "%"),
        fmt(session.get("mean_fps"), "{:.1f}"),
        status_text(session),
    ]


TABLE_HEADERS = [
    "Date", "Exercise", "Reps", "Peak", "Movement score", "DTW shape",
    "Scored movements", "Arm tracked", "FPS", "Status",
]

# Posture deviations above this (degrees) are physically implausible for
# a seated patient and almost certainly tracking or depth errors.
POSTURE_PLAUSIBLE_MAX_DEG = 45.0


def series(sessions, field):
    """(indices, values) for a field, skipping sessions without it."""

    xs, ys = [], []
    for index, session in enumerate(sessions, start=1):
        value = session.get(field)
        if value is not None:
            xs.append(index)
            ys.append(value)
    return xs, ys


def target_series(sessions, name):
    xs, ys = [], []
    for index, session in enumerate(sessions, start=1):
        raw = session.get("parameters_json")
        if raw:
            value = json.loads(raw).get(name)
            if value is not None:
                xs.append(index)
                ys.append(value)
    return xs, ys


def posture_text(session):
    """Experimental posture summary for therapist review, or None."""

    raw = session.get("experimental_posture_json")
    if not raw:
        return None

    data = json.loads(raw)
    header = (
        "EXPERIMENTAL posture measurements (not validated; not used for "
        "adaptation)"
    )

    if not data.get("measures"):
        reason = (
            "no resting baseline" if not data.get("baseline_established")
            else "no movement frames with trunk visible"
        )
        return f"{header}: unavailable ({reason})."

    parts = []
    for name, m in data["measures"].items():
        part = (f"{name.replace('_', ' ')}: p95 {m['p95_abs_deviation']:g} "
                f"{m['unit']}")
        if (
            m["unit"] == "degrees"
            and max(m["p95_abs_deviation"], m.get("max_abs_deviation", 0))
            > POSTURE_PLAUSIBLE_MAX_DEG
        ):
            part += " (implausible value: probably a tracking error)"
        parts.append(part)
    return f"{header}: " + "; ".join(parts) + "."


def compensation_summary_text(summary):
    """
    One-line text for a geometric compensation summary (dict), or None.
    Distinguishes detected / not detected / not assessed; never reports
    an unassessed pattern as "not detected".
    """

    if not summary:
        return None

    header = (
        "Compensation check (rule-based, provisional thresholds, not "
        "clinically validated; not used for adaptation)"
    )
    patterns = summary.get("patterns") or {}
    if not summary.get("baseline_established"):
        return (f"{header}: not assessed (no stable resting posture at the "
                "start of the session).")

    parts = []
    for p in patterns.values():
        name = p["label"]
        if not p["assessed"]:
            reasons = p.get("unable_reasons") or {}
            reason = max(reasons, key=reasons.get) if reasons else "no data"
            parts.append(f"{name}: not assessed ({reason})")
        elif p["episodes"]:
            parts.append(
                f"{name}: DETECTED in {100 * p['detected_fraction']:.0f}% "
                f"of assessed movement frames ({p['episodes']} episode(s); "
                f"max {p['max_abs_deviation']:g} {p['unit']}, threshold "
                f"{p['enter_threshold']:g})")
        else:
            parts.append(
                f"{name}: not detected (max {p['max_abs_deviation']:g} "
                f"{p['unit']}, threshold {p['enter_threshold']:g})")
    return f"{header}: " + "; ".join(parts) + "."


def compensation_text(session):
    """Stored compensation summary of a session row, or None."""

    raw = session.get("compensation_json")
    if not raw:
        return None
    try:
        return compensation_summary_text(json.loads(raw))
    except (TypeError, ValueError, KeyError):
        return "Compensation check: stored summary could not be read."
