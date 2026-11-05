"""Add administrator and therapist API accounts.

Revision ID: 0002_api_users
Revises: 0001_nars_initial
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_api_users"
down_revision = "0001_nars_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "api_users",
        sa.Column("user_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.Text(), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_api_users_username", "api_users", ["username"], unique=True)


def downgrade():
    op.drop_index("ix_api_users_username", table_name="api_users")
    op.drop_table("api_users")
