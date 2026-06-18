"""add gamification tables and total_xp column

Revision ID: g7h8i9j0k1l2
Revises: f6a7b8c9d0e1
Create Date: 2026-06-19 12:00:00.000000

Adds the gamification infrastructure for the Streak Heatmap and
XP / Leveling features:

  - ``login_events`` table — one row per successful login. The
    Dashboard heatmap aggregates from this table to render the
    last 12 weeks of login activity.

  - ``xp_events`` table — audit log of every XP award. Each row
    has a ``source`` discriminator (e.g. ``mastery_milestone``)
    and JSONB ``metadata`` for the per-source context (e.g. the
    milestone threshold and the topic that was mastered).

  - ``users.total_xp`` column — denormalized XP total. Lives on
    the user row for fast reads in the Dashboard's XP card and
    level-up checks. The ``xp_events`` table is the audit
    source-of-truth; ``total_xp`` is a cache.

No data backfill. Per the user's decision (Option A — fresh
start), existing users begin at 0 XP. Their accumulated mastery
scores are still in the database, so as they cross the milestone
thresholds on future activity (quiz, self-assessment, etc.)
they'll naturally earn XP from the new system.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB


# revision identifiers, used by Alembic.
#
# IMPORTANT — why we set down_revision to 'a1b2c3d4e5f6' (not
# 'f6a7b8c9d0e1' which is the previously-tip revision):
#
# Alembic's upgrade `head` resolves to "the most recent revision(s)
# not yet consumed by another revision". The two previous feature
# branches — the streak migration (a1b2c3d4e5f6) and this
# gamification migration — both branched from f6a7b8c9d0e1, so
# they were SIBLINGS. Without intervention, `alembic upgrade head`
# fails with "Multiple head revisions are present" because there is
# no linear path forward.
#
# By setting this migration's down_revision to a1b2c3d4e5f6 (the
# streak migration that was developed before this one), we collapse
# the two branches into a single chain:
#
#   f6a7b8c9d0e1 (concept_graph)
#     └─ a1b2c3d4e5f6 (streak)
#          └─ g7h8i9j0k1l2 (gamification)   ← this file
#
# Equivalent to a merge migration but doesn't require an extra
# empty file. If you're adding a new migration in the future,
# point its down_revision to 'g7h8i9j0k1l2' (or whichever is
# the new head after this fix).
revision = "g7h8i9j0k1l2"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── login_events ─────────────────────────────────────
    op.create_table(
        "login_events",
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "logged_in_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_login_events_user_logged_at",
        "login_events",
        ["user_id", "logged_in_at"],
    )

    # ── xp_events ────────────────────────────────────────
    op.create_table(
        "xp_events",
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_xp_events_user_created",
        "xp_events",
        ["user_id", "created_at"],
    )
    op.create_index(
        "ix_xp_events_user_source_created",
        "xp_events",
        ["user_id", "source", "created_at"],
    )

    # ── users.total_xp ───────────────────────────────────
    op.add_column(
        "users",
        sa.Column(
            "total_xp",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "total_xp")
    op.drop_index("ix_xp_events_user_source_created", table_name="xp_events")
    op.drop_index("ix_xp_events_user_created", table_name="xp_events")
    op.drop_table("xp_events")
    op.drop_index("ix_login_events_user_logged_at", table_name="login_events")
    op.drop_table("login_events")
