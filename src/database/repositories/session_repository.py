"""SQLAlchemy persistence for completed rehabilitation sessions."""

import json
from datetime import datetime

from sqlalchemy.orm import Session

from src.database import db
from src.database.models import RehabSession, SessionFrame


def _json_or_none(value):
    return None if value is None else json.dumps(value, sort_keys=True)


def _frame_values(frame):
    t_s, status, angle, plane, visibility, reference_frame = frame
    return {
        "t_s": None if t_s is None else round(float(t_s), 4),
        "status": status,
        "angle": None if angle is None else round(float(angle), 3),
        "plane": None if plane is None else round(float(plane), 2),
        "min_visibility": None if visibility is None else round(float(visibility), 3),
        "reference_frame": reference_frame,
    }


def save_session_summary(
    session: Session, *, patient_id: int, exercise_name: str, stats: dict,
    prescription_id: int | None = None, session_status: str | None = None,
):
    """Persist a session summary and all frame diagnostics in one transaction."""
    status = (
        session_status or stats.get("session_status")
        or db.SESSION_STATUS_COMPLETE
    )
    compensation = stats.get("compensation") or {}
    row = RehabSession(
        patient_id=patient_id,
        exercise=exercise_name,
        reps=stats["reps"],
        max_angle=stats.get("max_angle"),
        min_angle=stats.get("min_angle"),
        session_date=datetime.now(),
        exercise_id=stats.get("exercise_id"),
        affected_side=stats.get("side"),
        prescription_id=prescription_id,
        parameters_json=_json_or_none(stats.get("parameters")),
        metrics_version=stats.get("metrics_version") or db.METRICS_VERSION,
        target_reps=stats.get("target_reps"),
        off_plane_reps=stats.get("off_plane_reps"),
        trajectory_similarity=stats.get("trajectory_similarity"),
        valid_segments=stats.get("valid_segments"),
        mean_velocity_peaks=stats.get("mean_velocity_peaks"),
        segments_json=_json_or_none(stats.get("segments")),
        duration_s=stats.get("duration_s"),
        frames_total=stats.get("frames_total"),
        frames_with_pose=stats.get("frames_with_pose"),
        frames_measured=stats.get("frames_measured"),
        frame_errors=stats.get("frame_errors"),
        mean_fps=stats.get("mean_fps"),
        experimental_posture_json=_json_or_none(stats.get("experimental_posture")),
        compensation_method=compensation.get("method"),
        comp_windows_total=None,
        comp_no_compensation=None,
        comp_shoulder_elevation=None,
        comp_trunk_rotation=None,
        comp_lean_forward=None,
        peak_rom=stats.get("peak_rom"),
        dtw_similarity=stats.get("dtw_similarity"),
        dtw_mean_error_deg=stats.get("dtw_mean_error_deg"),
        session_status=status,
        interrupted_reps=stats.get("interrupted_reps"),
        diagnostics_json=_json_or_none(stats.get("diagnostics")),
        compensation_json=_json_or_none(compensation) if compensation else None,
    )
    session.add(row)
    session.flush()
    for index, frame in enumerate(stats.get("frame_log") or []):
        session.add(SessionFrame(
            session_id=row.session_id,
            frame_index=index,
            **_frame_values(frame),
        ))
    session.commit()
    session.refresh(row)
    return row.session_id
