"""merge heads 7d57968535ad and c3d4e5f6a7b8

Revision ID: d4e5f6a7b8c9
Revises: 7d57968535ad, c3d4e5f6a7b8
Create Date: 2026-06-14 12:00:00.000000

The codebase has two parallel migration branches that both need to be
considered as the head. This merge unifies them so a single `alembic
upgrade head` brings the database up to date.

After this merge:
  - DB is at d4e5f6a7b8c9 (the unified head)
  - agent_logs table is created
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd4e5f6a7b8c9'
down_revision = ('7d57968535ad', 'c3d4e5f6a7b8')
branch_labels = None
depends_on = None


def upgrade() -> None:
    # No schema changes — this is a pure merge migration.
    # The c3d4e5f6a7b8 branch already created agent_logs.
    pass


def downgrade() -> None:
    # Going back to either branch — let Alembic handle it via the
    # next-down revision chain.
    pass
