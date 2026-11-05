"""Unit tests for SQLAlchemy progress and dashboard reads."""

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, ExerciseRecommendation, Patient, RehabSession
from src.database.repositories import progress_repository


def test_progress_history_filters_and_preserves_legacy_null_metrics():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    patient = Patient(name="Progress Test Patient")
    session.add(patient)
    session.flush()
    current = RehabSession(
        patient_id=patient.patient_id,
        exercise="Shoulder Flexion",
        exercise_id="shoulder_flexion",
        affected_side="LEFT",
        reps=6,
        metrics_version=3,
        session_date=datetime.now(),
        session_status="COMPLETE",
    )
    legacy = RehabSession(
        patient_id=patient.patient_id,
        exercise="Legacy Exercise",
        reps=2,
        metrics_version=None,
        session_date=datetime.now() - timedelta(days=1),
    )
    session.add_all([current, legacy])
    session.commit()

    all_rows = progress_repository.list_patient_sessions(session, patient.patient_id)
    assert len(all_rows) == 2
    assert all_rows[0]["metrics_version"] is None
    assert all_rows[1]["metrics_version"] == 3

    filtered = progress_repository.list_patient_sessions(
        session, patient.patient_id, exercise_id="shoulder_flexion",
        affected_side="LEFT",
    )
    assert len(filtered) == 1
    assert filtered[0]["session_id"] == current.session_id

    session.close()
    engine.dispose()


def test_dashboard_counts_and_recent_sessions_are_database_derived():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    patient = Patient(name="Dashboard Test Patient")
    session.add(patient)
    session.flush()
    rehab = RehabSession(
        patient_id=patient.patient_id,
        exercise="Shoulder Flexion",
        reps=4,
        metrics_version=3,
        session_date=datetime.now(),
    )
    pending = ExerciseRecommendation(
        patient_id=patient.patient_id,
        assessment_id=None,
        exercise="Shoulder Flexion",
        difficulty="Level 1",
        status="PENDING",
        source="ADAPTATION",
    )
    # Assessment id is nullable in this isolated model fixture? The schema
    # requires it, so create a minimal assessment first.
    from src.database.models import ClinicalAssessment
    assessment = ClinicalAssessment(
        patient_id=patient.patient_id,
        diagnosis="Stroke",
        impairment="Reduced range",
    )
    session.add(assessment)
    session.flush()
    pending.assessment_id = assessment.assessment_id
    session.add_all([rehab, pending])
    session.commit()

    counts = progress_repository.dashboard_counts(session)
    assert counts["patients"] == 1
    assert counts["sessions"] == 1
    assert counts["sessions_last_7_days"] == 1
    assert counts["pending_adaptations"] == 1
    assert counts["last_session"] is not None

    recent = progress_repository.recent_sessions(session)
    assert len(recent) == 1
    assert recent[0]["patient_name"] == "Dashboard Test Patient"
    assert recent[0]["session_id"] == rehab.session_id

    session.close()
    engine.dispose()
