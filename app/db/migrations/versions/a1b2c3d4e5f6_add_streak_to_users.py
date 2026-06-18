"""add gamification streak fields to users

Revision ID: a1b2c3d4e5f6
Revises: f6a7b8c9d0e1
Create Date: 2026-06-18 09:00:00.000000

Adds three columns to the ``users`` table to support the daily login
streak gamification feature:

  - ``current_streak``  : Integer, default 0. Number of consecutive
                           days the user has logged in. Reset to 1
                           on login if more than 1 day has passed
                           since ``last_login_date``.
  - ``longest_streak``  : Integer, default 0. The longest streak the
                           user has ever achieved. Never decreases.
  - ``last_login_date`` : Date, nullable. The date of the most recent
                           login (used to compute the streak delta).

All three default to safe values so existing rows are valid after
the migration. ``last_login_date`` is nullable because we don't know
when pre-existing users last logged in — the next login will populate
it (and reset the streak to 1).
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "current_streak",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "longest_streak",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "users",
        sa.Column("last_login_date", sa.Date(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "last_login_date")
    op.drop_column("users", "longest_streak")
    op.drop_column("users", "current_streak")
