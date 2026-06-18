import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.database import get_db
from app.config import settings
from app.schemas.auth import TokenData
from app.models.user import User
from app.models.learning import LearningSession, Topic

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except jwt.PyJWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == token_data.user_id))
    user = result.scalars().first()
    if user is None:
        raise credentials_exception
    return user


def _to_uuid(value, field_name: str = "id") -> uuid.UUID:
    """Coerce a path/body value into a UUID, raising a clean 400 on bad input."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name} format",
        )


async def verify_session_owner(
    session_id,
    current_user: User,
    db: AsyncSession,
) -> LearningSession:
    """
    Ensure the given learning session exists AND belongs to ``current_user``.

    Returns the ``LearningSession`` row when ownership checks pass. Raises 404
    when the session is missing or belongs to someone else (we deliberately
    use 404 rather than 403 so a caller who doesn't own the resource learns
    nothing about its existence) and 400 when the id is malformed.
    """
    session_uuid = _to_uuid(session_id, "session_id")
    result = await db.execute(
        select(LearningSession).where(LearningSession.id == session_uuid)
    )
    session = result.scalars().first()
    if session is None or str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session not found",
        )
    return session


async def verify_topic_owner(
    topic_id: str,
    current_user: User,
    db: AsyncSession,
    *,
    require_session_id: uuid.UUID | None = None,
) -> Topic:
    """
    Ensure ``topic_id`` belongs to a session owned by ``current_user``.

    The lookup joins ``topics`` to ``learning_sessions`` and filters by the
    authenticated user, so a topic owned by another user is treated as
    "not found" (404) rather than forbidden — keeping the resource opaque to
    callers who don't own it.

    If ``require_session_id`` is provided, additionally assert the topic is
    scoped to that session (used when a request carries both a session_id and
    a topic_id that must agree). Returns the ``Topic`` row on success.
    """
    if not topic_id or not isinstance(topic_id, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid topic_id",
        )

    result = await db.execute(
        select(Topic)
        .join(LearningSession, Topic.session_id == LearningSession.id)
        .where(Topic.id == topic_id)
        .where(LearningSession.user_id == current_user.id)
    )
    topic = result.scalars().first()
    if topic is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Topic not found",
        )

    if require_session_id is not None and topic.session_id != require_session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Topic not found",
        )

    return topic
