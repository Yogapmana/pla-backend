"""LoginEvent — single row per successful login.

Used to render the streak heatmap on the Dashboard. We keep a
``Date``-granular audit trail (rather than relying on
``User.last_login_date`` which only stores the *most recent* date)
so we can compute the heatmap's per-day activity over the last
84-180 days.

A new row is inserted on every successful login (in
``streak_service.update_streak_on_login``). The ``logged_in_at``
timestamp is in the server's local timezone — see the comment on
``User.last_login_date`` for the multi-region caveat.

The composite index on ``(user_id, logged_in_at DESC)`` makes
the heatmap query (which is always ``WHERE user_id = ? AND
logged_in_at >= NOW() - INTERVAL 'N days'``) an O(log n) range
scan rather than a full table scan.
"""
import uuid
from sqlalchemy import Column, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


class LoginEvent(Base):
    __tablename__ = "login_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    logged_in_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Composite index — see module docstring.
    __table_args__ = (
        Index(
            "ix_login_events_user_logged_at",
            "user_id",
            "logged_in_at",
        ),
    )
