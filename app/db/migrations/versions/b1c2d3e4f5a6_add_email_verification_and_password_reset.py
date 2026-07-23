"""add_email_verification_and_password_reset

Revision ID: b1c2d3e4f5a6
Revises: 6c448d15ed61
Create Date: 2026-07-23
"""
from alembic import op
import sqlalchemy as sa

revision = "b1c2d3e4f5a6"
down_revision = "6c448d15ed61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_verified", sa.Boolean(), server_default="false", nullable=False),
    )
    op.add_column(
        "users",
        sa.Column("verification_code", sa.String(length=6), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("verification_code_expires", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("reset_token", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("reset_token_expires", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_users_verification_code",
        "users",
        ["verification_code"],
        unique=False,
    )
    op.create_index(
        "ix_users_reset_token",
        "users",
        ["reset_token"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_users_reset_token", table_name="users")
    op.drop_index("ix_users_verification_code", table_name="users")
    op.drop_column("users", "reset_token_expires")
    op.drop_column("users", "reset_token")
    op.drop_column("users", "verification_code_expires")
    op.drop_column("users", "verification_code")
    op.drop_column("users", "is_verified")
