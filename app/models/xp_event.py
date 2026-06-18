"""XpEvent — audit trail of every XP award.

Each row records a single XP-earning event. The ``source`` column
classifies the event type so we can render a per-source breakdown
in the XP history UI and so we can de-duplicate milestone XP
awards (e.g. "did this user already get the 0.5-mastery XP for
this topic?").

``amount`` is the XP delta awarded (always positive — XP only
grows). ``metadata`` carries context that the audit UI can show
the user, e.g. ``{"topic_id": "abc", "milestone": 0.5,
"mastery_score": 0.52}``.

The XP service writes these rows. The XP *total* lives on
``User.total_xp`` (denormalized for fast reads) and is the
authoritative number used by the level-up checks. ``xp_events``
is the immutable audit log.
"""
import uuid
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from app.db.database import Base


class XpEvent(Base):
    __tablename__ = "xp_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    # ``source`` is a short discriminator. Current values:
    #   - ``mastery_milestone``: crossed a 0.25/0.5/0.75/1.0 threshold
    #                             on a topic's mastery_score. The
    #                             specific threshold is in metadata.
    # Future sources (placeholders, not yet awarded):
    #   - ``daily_login``: a streak-day login bonus
    #   - ``first_topic``: first-time-per-topic bonus
    source = Column(String(50), nullable=False)
    amount = Column(Integer, nullable=False)
    # Free-form JSON. For ``mastery_milestone`` events this is
    # ``{"topic_id": str, "milestone": float, "mastery_score": float}``.
    # Other event types may use different shapes.
    metadata_json = Column("metadata", JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index(
            "ix_xp_events_user_created",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_xp_events_user_source_created",
            "user_id",
            "source",
            "created_at",
        ),
    )
