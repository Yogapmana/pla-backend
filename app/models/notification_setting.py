"""NotificationSetting — per-user notification preferences.

One row per user (user_id is the primary key). Persisted on the
server so preferences survive across devices/browsers, unlike the
old client-only localStorage approach.

Enforcement:
  - ``email_enabled`` (default True): gates transactional
    "updates" emails (progress + daily reminders). Auth-critical
    emails (verification, password reset) are never gated.
  - ``push_enabled``  (default True): gates in-app notifications
    created by ``NotificationService.create_notification``.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


class NotificationSetting(Base):
    __tablename__ = "notification_settings"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    email_enabled = Column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    push_enabled = Column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )