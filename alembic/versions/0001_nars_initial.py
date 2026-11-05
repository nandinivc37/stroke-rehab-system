"""Create the initial NARS relational schema.

This revision creates an empty target schema. Existing SQLite data is not
copied by this migration; use the separately reviewed migration utility.
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_nars_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "patients",
        sa.Column("patient_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("age", sa.Integer()),
        sa.Column("gender", sa.Text()),
        sa.Column("therapist", sa.Text()),
        sa.Column("notes", sa.Text()),
    )
    op.create_table(
        "affected_areas",
        sa.Column("area_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.Integer(), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("body_region", sa.Text(), nullable=False),
        sa.Column("joint", sa.Text()),
        sa.Column("notes", sa.Text()),
    )
    op.create_table(
        "clinical_assessments",
        sa.Column("assessment_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.Integer(), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("diagnosis", sa.Text(), nullable=False),
        sa.Column("impairment", sa.Text(), nullable=False),
        sa.Column("baseline_rom", sa.Float()),
        sa.Column("restrictions", sa.Text()),
        sa.Column("clinical_notes", sa.Text()),
        sa.Column("assessment_date", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_table(
        "exercise_recommendations",
        sa.Column("recommendation_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.Integer(), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("assessment_id", sa.Integer(), sa.ForeignKey("clinical_assessments.assessment_id"), nullable=False),
        sa.Column("exercise", sa.Text(), nullable=False),
        sa.Column("difficulty", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("approved_at", sa.DateTime()),
        sa.Column("exercise_id", sa.Text()),
        sa.Column("affected_side", sa.Text()),
        sa.Column("parameters_json", sa.Text()),
        sa.Column("source", sa.Text()),
        sa.Column("adaptation_decision_id", sa.Integer()),
        sa.Column("reviewed_by", sa.Text()),
        sa.Column("review_note", sa.Text()),
    )
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.Integer(), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("exercise", sa.Text()),
        sa.Column("reps", sa.Integer()),
        sa.Column("max_angle", sa.Float()),
        sa.Column("min_angle", sa.Float()),
        sa.Column("session_date", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("exercise_id", sa.Text()),
        sa.Column("affected_side", sa.Text()),
        sa.Column("prescription_id", sa.Integer()),
        sa.Column("parameters_json", sa.Text()),
        sa.Column("metrics_version", sa.Integer()),
        sa.Column("target_reps", sa.Integer()),
        sa.Column("off_plane_reps", sa.Integer()),
        sa.Column("trajectory_similarity", sa.Float()),
        sa.Column("valid_segments", sa.Integer()),
        sa.Column("mean_velocity_peaks", sa.Float()),
        sa.Column("segments_json", sa.Text()),
        sa.Column("duration_s", sa.Float()),
        sa.Column("frames_total", sa.Integer()),
        sa.Column("frames_with_pose", sa.Integer()),
        sa.Column("frames_measured", sa.Integer()),
        sa.Column("frame_errors", sa.Integer()),
        sa.Column("mean_fps", sa.Float()),
        sa.Column("experimental_posture_json", sa.Text()),
        sa.Column("compensation_method", sa.Text()),
        sa.Column("comp_windows_total", sa.Integer()),
        sa.Column("comp_no_compensation", sa.Integer()),
        sa.Column("comp_shoulder_elevation", sa.Integer()),
        sa.Column("comp_trunk_rotation", sa.Integer()),
        sa.Column("comp_lean_forward", sa.Integer()),
        sa.Column("peak_rom", sa.Float()),
        sa.Column("dtw_similarity", sa.Float()),
        sa.Column("dtw_mean_error_deg", sa.Float()),
        sa.Column("session_status", sa.Text()),
        sa.Column("interrupted_reps", sa.Integer()),
        sa.Column("diagnostics_json", sa.Text()),
        sa.Column("compensation_json", sa.Text()),
    )
    op.create_table(
        "session_frames",
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.session_id"), primary_key=True),
        sa.Column("frame_index", sa.Integer(), primary_key=True),
        sa.Column("t_s", sa.Float()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("angle", sa.Float()),
        sa.Column("plane", sa.Float()),
        sa.Column("min_visibility", sa.Float()),
        sa.Column("reference_frame", sa.Text()),
    )
    op.create_table(
        "adaptation_decisions",
        sa.Column("decision_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.Integer(), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("exercise_id", sa.Text(), nullable=False),
        sa.Column("affected_side", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.session_id")),
        sa.Column("policy_version", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("inputs_json", sa.Text(), nullable=False),
        sa.Column("previous_parameters_json", sa.Text()),
        sa.Column("proposed_parameters_json", sa.Text()),
        sa.Column("previous_recommendation_id", sa.Integer()),
        sa.Column("proposed_recommendation_id", sa.Integer()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_sessions_patient_date", "sessions", ["patient_id", "session_date"])
    op.create_index("ix_assessments_patient_date", "clinical_assessments", ["patient_id", "assessment_date"])
    op.create_index("ix_recommendations_patient_status", "exercise_recommendations", ["patient_id", "status"])
    op.create_index("ix_adaptation_patient_created", "adaptation_decisions", ["patient_id", "created_at"])


def downgrade():
    op.drop_index("ix_adaptation_patient_created", table_name="adaptation_decisions")
    op.drop_index("ix_recommendations_patient_status", table_name="exercise_recommendations")
    op.drop_index("ix_assessments_patient_date", table_name="clinical_assessments")
    op.drop_index("ix_sessions_patient_date", table_name="sessions")
    op.drop_table("adaptation_decisions")
    op.drop_table("session_frames")
    op.drop_table("sessions")
    op.drop_table("exercise_recommendations")
    op.drop_table("clinical_assessments")
    op.drop_table("affected_areas")
    op.drop_table("patients")
