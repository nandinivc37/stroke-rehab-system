"""ORM mappings for the NARS clinical and rehabilitation data schema.

These mappings describe the shared SQLite/PostgreSQL schema during the staged
migration. Patient, affected-area, clinical-assessment and recommendation API
persistence now use SQLAlchemy repositories against the existing SQLite file.
Other application workflows remain on the legacy SQLite data-access layer.
Preserve NULL values for historical measurements that were not recorded.
"""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database.sqlalchemy_support import Base


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    age: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(Text)
    therapist: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)


class AffectedArea(Base):
    __tablename__ = "affected_areas"

    area_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    body_region: Mapped[str] = mapped_column(Text, nullable=False)
    joint: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)


class ClinicalAssessment(Base):
    __tablename__ = "clinical_assessments"

    assessment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    diagnosis: Mapped[str] = mapped_column(Text, nullable=False)
    impairment: Mapped[str] = mapped_column(Text, nullable=False)
    baseline_rom: Mapped[float | None] = mapped_column(Float)
    restrictions: Mapped[str | None] = mapped_column(Text)
    clinical_notes: Mapped[str | None] = mapped_column(Text)
    assessment_date: Mapped[datetime | None] = mapped_column(DateTime)


class ExerciseRecommendation(Base):
    __tablename__ = "exercise_recommendations"

    recommendation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("clinical_assessments.assessment_id"), nullable=False)
    exercise: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="PENDING")
    created_at: Mapped[datetime | None] = mapped_column(DateTime)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime)
    exercise_id: Mapped[str | None] = mapped_column(Text)
    affected_side: Mapped[str | None] = mapped_column(Text)
    parameters_json: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)
    adaptation_decision_id: Mapped[int | None] = mapped_column(Integer)
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    review_note: Mapped[str | None] = mapped_column(Text)


class RehabSession(Base):
    __tablename__ = "sessions"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    exercise: Mapped[str | None] = mapped_column(Text)
    reps: Mapped[int | None] = mapped_column(Integer)
    max_angle: Mapped[float | None] = mapped_column(Float)
    min_angle: Mapped[float | None] = mapped_column(Float)
    session_date: Mapped[datetime | None] = mapped_column(DateTime)
    exercise_id: Mapped[str | None] = mapped_column(Text)
    affected_side: Mapped[str | None] = mapped_column(Text)
    prescription_id: Mapped[int | None] = mapped_column(Integer)
    parameters_json: Mapped[str | None] = mapped_column(Text)
    metrics_version: Mapped[int | None] = mapped_column(Integer)
    target_reps: Mapped[int | None] = mapped_column(Integer)
    off_plane_reps: Mapped[int | None] = mapped_column(Integer)
    trajectory_similarity: Mapped[float | None] = mapped_column(Float)
    valid_segments: Mapped[int | None] = mapped_column(Integer)
    mean_velocity_peaks: Mapped[float | None] = mapped_column(Float)
    segments_json: Mapped[str | None] = mapped_column(Text)
    duration_s: Mapped[float | None] = mapped_column(Float)
    frames_total: Mapped[int | None] = mapped_column(Integer)
    frames_with_pose: Mapped[int | None] = mapped_column(Integer)
    frames_measured: Mapped[int | None] = mapped_column(Integer)
    frame_errors: Mapped[int | None] = mapped_column(Integer)
    mean_fps: Mapped[float | None] = mapped_column(Float)
    experimental_posture_json: Mapped[str | None] = mapped_column(Text)
    compensation_method: Mapped[str | None] = mapped_column(Text)
    comp_windows_total: Mapped[int | None] = mapped_column(Integer)
    comp_no_compensation: Mapped[int | None] = mapped_column(Integer)
    comp_shoulder_elevation: Mapped[int | None] = mapped_column(Integer)
    comp_trunk_rotation: Mapped[int | None] = mapped_column(Integer)
    comp_lean_forward: Mapped[int | None] = mapped_column(Integer)
    peak_rom: Mapped[float | None] = mapped_column(Float)
    dtw_similarity: Mapped[float | None] = mapped_column(Float)
    dtw_mean_error_deg: Mapped[float | None] = mapped_column(Float)
    session_status: Mapped[str | None] = mapped_column(Text)
    interrupted_reps: Mapped[int | None] = mapped_column(Integer)
    diagnostics_json: Mapped[str | None] = mapped_column(Text)
    compensation_json: Mapped[str | None] = mapped_column(Text)


class SessionFrame(Base):
    __tablename__ = "session_frames"

    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.session_id"), primary_key=True)
    frame_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    t_s: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    angle: Mapped[float | None] = mapped_column(Float)
    plane: Mapped[float | None] = mapped_column(Float)
    min_visibility: Mapped[float | None] = mapped_column(Float)
    reference_frame: Mapped[str | None] = mapped_column(Text)


class AdaptationDecision(Base):
    __tablename__ = "adaptation_decisions"

    decision_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    exercise_id: Mapped[str] = mapped_column(Text, nullable=False)
    affected_side: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.session_id"))
    policy_version: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    inputs_json: Mapped[str] = mapped_column(Text, nullable=False)
    previous_parameters_json: Mapped[str | None] = mapped_column(Text)
    proposed_parameters_json: Mapped[str | None] = mapped_column(Text)
    previous_recommendation_id: Mapped[int | None] = mapped_column(Integer)
    proposed_recommendation_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime | None] = mapped_column(DateTime)



class PatientTherapistAssignment(Base):
    """Explicit therapist-to-patient access grant."""

    __tablename__ = "patient_therapist_assignments"

    assignment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    therapist_user_id: Mapped[int] = mapped_column(ForeignKey("api_users.user_id"), nullable=False)
    assigned_by_user_id: Mapped[int] = mapped_column(ForeignKey("api_users.user_id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())


class LiveSession(Base):
    """Durable lifecycle metadata; live analyzer state remains process-local."""

    __tablename__ = "live_sessions"

    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    prescription_id: Mapped[int] = mapped_column(Integer, nullable=False)
    exercise_id: Mapped[str] = mapped_column(Text, nullable=False)
    affected_side: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    frames_received: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ApiUser(Base):
    __tablename__ = "api_users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(Text, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime | None] = mapped_column(DateTime)


ApiUser.__table__.append_constraint(
    CheckConstraint("role IN ('therapist', 'admin')", name="ck_api_users_role")
)
Index("uq_api_users_username_lower", func.lower(ApiUser.username), unique=True)


Index("uq_patient_therapist_assignment", PatientTherapistAssignment.patient_id, PatientTherapistAssignment.therapist_user_id, unique=True)
Index("ix_patient_assignment_therapist", PatientTherapistAssignment.therapist_user_id)

Index("ix_sessions_patient_date", RehabSession.patient_id, RehabSession.session_date)
Index("ix_assessments_patient_date", ClinicalAssessment.patient_id, ClinicalAssessment.assessment_date)
Index("ix_recommendations_patient_status", ExerciseRecommendation.patient_id, ExerciseRecommendation.status)
Index("ix_adaptation_patient_created", AdaptationDecision.patient_id, AdaptationDecision.created_at)
