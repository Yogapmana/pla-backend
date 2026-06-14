"""add agent_logs table

Revision ID: c3d4e5f6a7b8
Revises: 7d57968535ad
Create Date: 2026-06-14 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB


# revision identifiers, used by Alembic.
revision = 'c3d4e5f6a7b8'
down_revision = 'a6e556951e36'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_logs",
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("learning_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("agent", sa.String(length=50), nullable=False),
        sa.Column("level", sa.String(length=20), nullable=False, server_default="info"),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_logs_session_id", "agent_logs", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_agent_logs_session_id", table_name="agent_logs")
    op.drop_table("agent_logs")
