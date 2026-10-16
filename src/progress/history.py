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


def table_row(session):
    """Display strings for one session row."""

    unit = unit_for(session)
    return [
        session.get("session_date") or MISSING,
        short_label(session),
        reps_text(session),
        fmt(session.get("max_angle"), suffix=f" {unit}".rstrip()),
        fmt(session.get("trajectory_similarity"), suffix="%"),
        fmt(session.get("valid_segments"), "{:d}"),
        fmt(pose_coverage(session), "{:.0f}", "%"),
        fmt(session.get("mean_fps"), "{:.1f}"),
    ]


TABLE_HEADERS = [
    "Date", "Exercise", "Reps", "Peak", "Similarity", "Attempts",
    "Arm tracked", "FPS",
]


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

    parts = [
        f"{name.replace('_', ' ')}: p95 {m['p95_abs_deviation']:g} {m['unit']}"
        for name, m in data["measures"].items()
    ]
    return f"{header}: " + "; ".join(parts) + "."
