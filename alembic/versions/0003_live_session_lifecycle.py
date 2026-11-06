"""Add durable live-session lifecycle metadata."""

from alembic import op
import sqlalchemy as sa

revision = "0003_live_session_lifecycle"
down_revision = "0002_api_users"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "live_sessions",
        sa.Column("session_id", sa.Text(), primary_key=True),
        sa.Column("patient_id", sa.Integer(), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("prescription_id", sa.Integer(), nullable=False),
        sa.Column("exercise_id", sa.Text(), nullable=False),
        sa.Column("affected_side", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime()),
        sa.Column("frames_received", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_live_sessions_patient_started", "live_sessions", ["patient_id", "started_at"])
    op.create_index("ix_live_sessions_status", "live_sessions", ["status"])


def downgrade():
    op.drop_index("ix_live_sessions_status", table_name="live_sessions")
    op.drop_index("ix_live_sessions_patient_started", table_name="live_sessions")
    op.drop_table("live_sessions")
