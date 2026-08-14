from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Response
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.refresh_token import RefreshToken

ALGORITHM = "HS256"

ACCESS_COOKIE = "pla_access"
REFRESH_COOKIE = "pla_refresh"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    to_encode = data.copy()
    expire = _utcnow() + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str, *, expected_type: str | None = None) -> dict[str, Any]:
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    if expected_type:
        token_type = payload.get("type")
        # Legacy access tokens (pre-rotation) have no `type` claim — accept as access only.
        if token_type is None and expected_type == "access":
            return payload
        if token_type != expected_type:
            raise jwt.InvalidTokenError(f"Expected token type {expected_type}")
    return payload


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def create_refresh_token(
    db: AsyncSession,
    user_id,
    *,
    user_agent: str | None = None,
) -> str:
    """Create a refresh JWT, persist its hash, return the raw token."""
    jti = secrets.token_urlsafe(32)
    expires = _utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "jti": jti,
        "type": "refresh",
        "exp": expires,
    }
    raw = jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)

    row = RefreshToken(
        user_id=user_id,
        jti=jti,
        token_hash=hash_token(raw),
        expires_at=expires,
        user_agent=(user_agent or "")[:512] or None,
    )
    db.add(row)
    await db.flush()
    return raw


async def rotate_refresh_token(
    db: AsyncSession,
    raw_refresh: str,
    *,
    user_agent: str | None = None,
) -> tuple[str, str]:
    """
    Validate refresh token, revoke it, issue new access + refresh pair.
    Returns (access_token, new_refresh_token).
    """
    try:
        payload = decode_token(raw_refresh, expected_type="refresh")
    except jwt.PyJWTError as e:
        raise ValueError("Invalid refresh token") from e

    user_id = payload.get("sub")
    jti = payload.get("jti")
    if not user_id or not jti:
        raise ValueError("Invalid refresh token payload")

    result = await db.execute(
        select(RefreshToken).where(RefreshToken.jti == jti)
    )
    stored = result.scalars().first()
    if stored is None:
        raise ValueError("Refresh token not found")
    if stored.revoked_at is not None:
        # Possible reuse after theft — revoke all sessions for this user
        await revoke_all_user_refresh_tokens(db, stored.user_id)
        raise ValueError("Refresh token already revoked")
    if stored.expires_at < _utcnow():
        raise ValueError("Refresh token expired")
    if stored.token_hash != hash_token(raw_refresh):
        raise ValueError("Refresh token mismatch")

    stored.revoked_at = _utcnow()
    new_refresh = await create_refresh_token(
        db, stored.user_id, user_agent=user_agent
    )
    access = create_access_token(data={"sub": str(stored.user_id)})
    return access, new_refresh


async def revoke_refresh_token(db: AsyncSession, raw_refresh: str | None) -> None:
    if not raw_refresh:
        return
    try:
        payload = decode_token(raw_refresh, expected_type="refresh")
    except jwt.PyJWTError:
        return
    jti = payload.get("jti")
    if not jti:
        return
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.jti == jti, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=_utcnow())
    )


async def revoke_all_user_refresh_tokens(db: AsyncSession, user_id) -> None:
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=_utcnow())
    )


def _cookie_secure() -> bool:
    """Secure flag for cookies.

    Prefer explicit COOKIE_SECURE from env. Otherwise only enable Secure
    when *no* localhost/http origin is configured (pure HTTPS deploy).
    Mixed CORS lists (prod HTTPS + local HTTP) must stay Secure=False
    so local dev can store cookies.
    """
    explicit = (getattr(settings, "COOKIE_SECURE", None) or "").strip().lower()
    if explicit in ("1", "true", "yes"):
        return True
    if explicit in ("0", "false", "no"):
        return False
    origins = [o.strip() for o in (settings.CORS_ORIGINS or "").split(",") if o.strip()]
    if not origins:
        return False
    # Any http:// or localhost origin → not secure (local / mixed deploy)
    for o in origins:
        if o.startswith("http://") or "localhost" in o or "127.0.0.1" in o:
            return False
    return all(o.startswith("https://") for o in origins)


def _cookie_samesite() -> str:
    """SameSite for auth cookies. Production cross-site frontends (vercel.app)
    need "none" + Secure; same-site deploys can stay "lax"."""
    samesite = (getattr(settings, "COOKIE_SAMESITE", None) or "lax").strip().lower()
    return "none" if samesite == "none" else "lax"


def set_auth_cookies(
    response: Response,
    *,
    access_token: str,
    refresh_token: str,
) -> None:
    secure = _cookie_secure()
    samesite = _cookie_samesite()
    # Access: short-lived, sent on every API call
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=access_token,
        httponly=True,
        secure=secure,
        samesite=samesite,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    # Refresh: longer-lived, only needed by /auth/refresh + logout
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=refresh_token,
        httponly=True,
        secure=secure,
        samesite=samesite,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/api/v1/auth",
    )


def clear_auth_cookies(response: Response) -> None:
    secure = _cookie_secure()
    samesite = _cookie_samesite()
    response.delete_cookie(
        key=ACCESS_COOKIE, path="/", secure=secure, httponly=True, samesite=samesite
    )
    response.delete_cookie(
        key=REFRESH_COOKIE,
        path="/api/v1/auth",
        secure=secure,
        httponly=True,
        samesite=samesite,
    )


async def issue_token_pair(
    db: AsyncSession,
    user_id,
    *,
    user_agent: str | None = None,
) -> tuple[str, str]:
    """Return (access_token, refresh_token) and persist refresh hash."""
    access = create_access_token(data={"sub": str(user_id)})
    refresh = await create_refresh_token(db, user_id, user_agent=user_agent)
    return access, refresh
