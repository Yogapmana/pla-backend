"""add mindmap_json column to curricula

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-06-15 10:00:00.000000

Adds a nullable JSONB column ``mindmap_json`` to the ``curricula`` table to
cache the AI-generated Mermaid mind map for a learning session.

Schema::

    mindmap_json = {
        "syntax":     "<mermaid mindmap source>",   # Mermaid v11 syntax
        "summary":    "<one-paragraph summary>",     # LLM's verbal summary
        "generated_at": "<ISO8601>",
        "model":      "<model identifier>",
        "node_count": <int>,
    }

The column is nullable so existing curricula don't need a back-fill; the
mind map is generated on first request and cached thereafter.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = 'e5f6a7b8c9d0'
down_revision = 'd4e5f6a7b8c9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'curricula',
        sa.Column('mindmap_json', JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column('curricula', 'mindmap_json')
