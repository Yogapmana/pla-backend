import uuid
from sqlalchemy import Column, String, DateTime, Integer, Date
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ── Gamification: daily login streak ────────────────────────
    # The streak is the number of consecutive days the user has
    # logged in. Reset to 1 on a login that follows a >1 day gap.
    # See `app/services/streak_service.py` for the update logic.
    # `last_login_date` is in the SERVER's local timezone — the
    # spec assumes single-region deployment. For a multi-region
    # deploy, this should move to UTC and the client should send
    # its IANA timezone to disambiguate "today".
    current_streak = Column(
        Integer, nullable=False, default=0, server_default="0"
    )
    longest_streak = Column(
        Integer, nullable=False, default=0, server_default="0"
    )
    last_login_date = Column(Date, nullable=True)

    # ── Gamification: total XP ────────────────────────────────
    # Denormalized XP total — fast O(1) read in the Dashboard's
    # XP card. The authoritative audit log is the `xp_events`
    # table. Updated by `app/services/xp_service.py` whenever a
    # new XP event is awarded. Never decreases.
    #
    # We chose Option A (fresh start) for the rollout: existing
    # users begin at 0 XP and earn XP from the new milestone
    # system as they progress. Their previous mastery scores are
    # untouched — XP will accumulate naturally as they cross the
    # 0.25/0.5/0.75/1.0 thresholds on future activity.
    total_xp = Column(
        Integer, nullable=False, default=0, server_default="0"
    )
