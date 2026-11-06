"""Add explicit therapist-to-patient access assignments."""

from alembic import op
import sqlalchemy as sa

revision = "0004_patient_therapist_assignments"
down_revision = "0003_live_session_lifecycle"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "patient_therapist_assignments",
        sa.Column("assignment_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.Integer(), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("therapist_user_id", sa.Integer(), sa.ForeignKey("api_users.user_id"), nullable=False),
        sa.Column("assigned_by_user_id", sa.Integer(), sa.ForeignKey("api_users.user_id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("patient_id", "therapist_user_id", name="uq_patient_therapist_assignment"),
    )
    op.create_index(
        "ix_patient_assignment_therapist",
        "patient_therapist_assignments",
        ["therapist_user_id"],
    )


def downgrade():
    op.drop_index("ix_patient_assignment_therapist", table_name="patient_therapist_assignments")
    op.drop_table("patient_therapist_assignments")
