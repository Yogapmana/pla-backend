"""add concept_graph_json column to curricula

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-06-15 12:00:00.000000

Adds a nullable JSONB column ``concept_graph_json`` to the ``curricula`` table to
cache the structured concept graph (root → clusters → concepts → topics → resources)
for the interactive mind map view.

Schema::

    concept_graph_json = {
        "version":        1,                       # cache schema version
        "version_marker": <int>,                   # curriculum.version at build time
        "course_title":   "<string>",
        "generated_at":   "<ISO8601>",
        "model":          "<model name | 'fallback'>",
        "build_seconds":  <float>,
        "nodes": [...],                             # React Flow-shaped node dicts
        "edges": [...],                             # React Flow-shaped edge dicts
    }

The column is nullable so existing curricula don't need a back-fill; the graph
is generated on first request and cached thereafter. ``version_marker`` enables
self-healing cache invalidation: if the curriculum's ``version`` is bumped
(e.g. after a replan) without the cache being cleared, the read path detects
the mismatch and rebuilds.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = 'f6a7b8c9d0e1'
down_revision = 'e5f6a7b8c9d0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'curricula',
        sa.Column('concept_graph_json', JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column('curricula', 'concept_graph_json')
