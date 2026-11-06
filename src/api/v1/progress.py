"""Read-only patient progress and therapist dashboard endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session, get_current_user, require_patient_access
from src.services import progress_service
from src.services.patient_service import PatientNotFoundError

router = APIRouter(tags=["progress and dashboard"])


@router.get("/dashboard/summary")
def dashboard_summary(database_session: Session = Depends(get_db_session)):
    return progress_service.dashboard_summary(database_session)


@router.get("/patients/{patient_id}/progress")
def patient_progress(
    patient_id: int,
    exercise_id: str | None = None,
    affected_side: str | None = None,
    database_session: Session = Depends(get_db_session),
    _access=Depends(require_patient_access),
):
    try:
        return progress_service.patient_progress(
            patient_id,
            exercise_id=exercise_id,
            affected_side=affected_side,
            database_session=database_session,
        )
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/patients/{patient_id}/progress/trends")
def patient_progress_trends(
    patient_id: int,
    exercise_id: str | None = None,
    affected_side: str | None = None,
    database_session: Session = Depends(get_db_session),
    _access=Depends(require_patient_access),
):
    try:
        result = progress_service.patient_progress(
            patient_id,
            exercise_id=exercise_id,
            affected_side=affected_side,
            database_session=database_session,
        )
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "patient_id": patient_id,
        "groups": [
            {
                "exercise": group["exercise"],
                "affected_side": group["affected_side"],
                "legacy_metrics": group["legacy_metrics"],
                "metrics_version": group["metrics_version"],
                "label": group["label"],
                "points": [
                    {
                        "session_id": session["session_id"],
                        "session_date": session["session_date"],
                        "reps": session["reps"],
                        "peak_value": session["peak_value"],
                        "movement_score": session["trajectory_similarity"],
                        "dtw_shape": session["dtw_similarity"],
                        "pose_coverage_percent": session["pose_coverage_percent"],
                    }
                    for session in group["sessions"]
                ],
            }
            for group in result["groups"]
        ],
    }
