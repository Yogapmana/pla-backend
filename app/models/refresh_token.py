"""RefreshToken — server-side store for rotating refresh tokens.

Access tokens stay short-lived (JWT, not stored). Refresh tokens are
opaque jti values whose SHA-256 hash is stored here so we can revoke
them on logout / rotation without keeping the raw token.
"""
import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Public id embedded in the JWT refresh token (`jti` claim).
    jti = Column(String(64), unique=True, nullable=False, index=True)
    # SHA-256 hex of the raw refresh JWT — never store the raw token.
    token_hash = Column(String(64), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    user_agent = Column(String(512), nullable=True)

    __table_args__ = (
        Index("ix_refresh_tokens_user_id", "user_id"),
        Index("ix_refresh_tokens_user_expires", "user_id", "expires_at"),
    )
