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
    pass

def downgrade() -> None:
    pass
