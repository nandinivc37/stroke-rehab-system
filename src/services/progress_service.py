"""Read-only progress and dashboard services using existing metric helpers."""

from src.database import db
from src.database.repositories import progress_repository
from src.progress.history import (
    compensation_text,
    group_sessions,
    peak_value,
    pose_coverage,
    posture_text,
    short_label,
    status_text,
)
from src.services.patient_service import get_patient


def _session_read(row):
    return {
        "session_id": row.get("session_id"),
        "session_date": row.get("session_date"),
        "exercise_id": row.get("exercise_id"),
        "exercise": short_label(row),
        "metrics_version": row.get("metrics_version"),
        "affected_side": row.get("affected_side"),
        "session_status": status_text(row),
        "reps": row.get("reps"),
        "target_reps": row.get("target_reps"),
        "peak_value": peak_value(row),
        "trajectory_similarity": row.get("trajectory_similarity"),
        "dtw_similarity": row.get("dtw_similarity"),
        "valid_segments": row.get("valid_segments"),
        "pose_coverage_percent": pose_coverage(row),
        "mean_fps": row.get("mean_fps"),
        "parameters_json": row.get("parameters_json"),
        "compensation_summary": compensation_text(row),
        "experimental_posture_summary": posture_text(row),
    }


def patient_progress(patient_id, exercise_id=None, affected_side=None, database_session=None):
    get_patient(patient_id, session=database_session)
    if database_session is None:
        sessions = db.get_sessions(
            patient_id, exercise_id=exercise_id, affected_side=affected_side
        )
    else:
        sessions = progress_repository.list_patient_sessions(
            database_session, patient_id,
            exercise_id=exercise_id, affected_side=affected_side,
        )
    groups = group_sessions(sessions)
    grouped = []
    for key, rows in groups.items():
        grouped.append({
            "exercise": key.exercise,
            "affected_side": key.side or None,
            "legacy_metrics": key.legacy,
            "metrics_version": key.version,
            "label": key.label(),
            "sessions": [_session_read(row) for row in rows],
        })
    return {
        "patient_id": patient_id,
        "session_count": len(sessions),
        "groups": grouped,
        "sessions": [_session_read(row) for row in reversed(sessions)],
    }


def dashboard_summary(database_session=None):
    if database_session is None:
        counts = db.get_dashboard_counts()
    else:
        counts = progress_repository.dashboard_counts(database_session)
    if database_session is not None:
        recent = [
            {**row, **_session_read(row)}
            for row in progress_repository.recent_sessions(database_session, limit=10)
        ]
    else:
        patients = db.get_all_patients()
        recent = []
        for patient in patients:
            patient_id = patient[0]
            for session in db.get_sessions(patient_id)[-5:]:
                recent.append({
                    "patient_id": patient_id,
                    "patient_name": patient[1],
                    **_session_read(session),
                })
        recent.sort(
            key=lambda row: (row.get("session_date") or "", row.get("session_id") or 0),
            reverse=True,
        )
    return {
        "counts": counts,
        "recent_sessions": recent[:10],
    }
