import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel, EmailStr
from app.db.database import get_db
from app.schemas.auth import UserCreate, UserResponse, Token, UserLanguageUpdate, UserProfileUpdate
from app.models.user import User
from app.services.auth_service import (
    get_password_hash,
    verify_password,
    create_access_token,
    issue_token_pair,
    set_auth_cookies,
    clear_auth_cookies,
    rotate_refresh_token,
    revoke_refresh_token,
    revoke_all_user_refresh_tokens,
    REFRESH_COOKIE,
)
from app.services.streak_service import update_streak_on_login
from app.dependencies import get_current_user
from app.config import settings
from app.tasks.email_tasks import (
    send_welcome_email_task,
    send_verification_email_task,
    send_password_reset_email_task,
)
from google.oauth2 import id_token
from google.auth.transport import requests
import uuid

router = APIRouter()


def _generate_verification_code() -> str:
    """Generate a 6-digit numeric OTP code."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _generate_reset_token() -> str:
    """Generate a URL-safe random token for password reset."""
    return secrets.token_urlsafe(48)


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_in.email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user_in.password)
    code = _generate_verification_code()
    code_expires = datetime.now(timezone.utc) + timedelta(minutes=15)

    new_user = User(
        email=user_in.email,
        username=user_in.username,
        hashed_password=hashed_password,
        language_preference=user_in.language_preference,
        verification_code=code,
        verification_code_expires=code_expires,
        is_verified=False,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Send verification email asynchronously
    send_verification_email_task.delay(new_user.email, new_user.username, code)

    # Create welcome notification
    from app.services.notification_service import NotificationService
    notif_service = NotificationService(db)
    await notif_service.create_notification(
        user_id=new_user.id,
        title="Selamat Datang di Synapsa!",
        message="Mari mulai belajar hari ini. Buat kurikulum pertama Anda sekarang.",
        notification_type="welcome",
        link="/dashboard"
    )

    # Register issues tokens only after email verification — return user without session cookies.
    # Client must login after verify. Keep empty token for API compatibility.
    return {
        "access_token": "",
        "token_type": "bearer",
        "user": new_user,
        "streak": None,
    }


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str


@router.post("/verify-email")
async def verify_email(request: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    """Verify email with 6-digit OTP code."""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if user.is_verified:
        return {"message": "Email already verified"}

    if (
        not user.verification_code
        or not user.verification_code_expires
        or user.verification_code != request.code
    ):
        raise HTTPException(status_code=400, detail="Invalid verification code")

    if datetime.now(timezone.utc) > user.verification_code_expires:
        raise HTTPException(status_code=400, detail="Verification code has expired")

    user.is_verified = True
    user.verification_code = None
    user.verification_code_expires = None
    await db.commit()

    return {"message": "Email verified successfully"}


class ResendVerificationRequest(BaseModel):
    email: EmailStr


@router.post("/resend-verification")
async def resend_verification(request: ResendVerificationRequest, db: AsyncSession = Depends(get_db)):
    """Resend a new verification OTP code."""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if user.is_verified:
        return {"message": "Email already verified"}

    code = _generate_verification_code()
    user.verification_code = code
    user.verification_code_expires = datetime.now(timezone.utc) + timedelta(minutes=15)
    await db.commit()

    send_verification_email_task.delay(user.email, user.username, code)
    return {"message": "Verification code sent"}


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Send a password reset link to the user's email."""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalars().first()
    if not user:
        # Don't reveal whether email exists
        return {"message": "If the email exists, a reset link has been sent"}

    if user.auth_provider == "google":
        return {"message": "If the email exists, a reset link has been sent"}

    token = _generate_reset_token()
    user.reset_token = token
    user.reset_token_expires = datetime.now(timezone.utc) + timedelta(hours=1)
    await db.commit()

    reset_url = f"{settings.CORS_ORIGINS.split(',')[0]}/reset-password?token={token}"
    send_password_reset_email_task.delay(user.email, user.username, reset_url)

    return {"message": "If the email exists, a reset link has been sent"}


class ResetPasswordRequest(BaseModel):
    token: str
    password: str


@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Reset password using the token from email."""
    result = await db.execute(select(User).where(User.reset_token == request.token))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    if datetime.now(timezone.utc) > user.reset_token_expires:
        raise HTTPException(status_code=400, detail="Reset token has expired")

    user.hashed_password = get_password_hash(request.password)
    user.reset_token = None
    user.reset_token_expires = None
    await db.commit()

    return {"message": "Password reset successfully"}


@router.post("/login", response_model=Token)
async def login(
    response: Response,
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalars().first()
    if not user or not user.hashed_password or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check email verification for local auth users
    if user.auth_provider == "local" and not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your inbox for the verification code.",
        )

    # Update the user's daily login streak. The service mutates
    # the user in place and returns metadata (new_streak,
    # is_new_day, milestone) that the frontend uses to decide
    # whether to show a celebration modal.
    streak_data = await update_streak_on_login(db, user)
    access_token, refresh_token = await issue_token_pair(
        db, user.id, user_agent=request.headers.get("user-agent")
    )
    # Commit the streak update + refresh token row together.
    await db.commit()
    await db.refresh(user)

    set_auth_cookies(response, access_token=access_token, refresh_token=refresh_token)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
        "streak": streak_data,
    }


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    # No streak update on /me — this endpoint just reads the
    # current state. Streak updates happen ONLY on login/register
    # to avoid double-counting refresh-driven page loads.
    return current_user


@router.put("/me/language", response_model=UserResponse)
async def update_user_language(
    update_data: UserLanguageUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update the user's language preference and cascade to all learning sessions."""
    current_user.language_preference = update_data.language_preference
    
    # Cascade the language update to all of the user's learning sessions
    from app.models.learning import LearningSession
    from sqlalchemy import update
    await db.execute(
        update(LearningSession)
        .where(LearningSession.user_id == current_user.id)
        .values(language=update_data.language_preference)
    )
    
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_user_profile(
    update_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the current user's profile fields (username)."""
    username = (update_data.username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if len(username) < 2 or len(username) > 100:
        raise HTTPException(
            status_code=400,
            detail="Username must be between 2 and 100 characters",
        )

    current_user.username = username
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_account(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently delete the current user's account and related data."""
    # `users` cascades to `learning_sessions`, but the session-level tables
    # (learning_modules / resource_links) reference sessions with NO ACTION
    # foreign keys — deleting the user directly would raise a FK violation.
    # Purge every session (and its non-cascading children) first.
    from app.services.learning_service import LearningService
    learning_service = LearningService(db)
    await learning_service.purge_user_sessions(current_user.id)

    # The remaining user-owned rows (notifications, xp_events,
    # login_events, refresh_tokens, notification_settings) all cascade.
    await revoke_all_user_refresh_tokens(db, current_user.id)
    await db.delete(current_user)
    await db.commit()
    clear_auth_cookies(response)
    return None


@router.post("/refresh", response_model=Token)
async def refresh_token_endpoint(
    response: Response,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Rotate refresh token (httpOnly cookie) and issue a new access token.

    Does NOT require a valid access token — only the refresh cookie.
    On reuse of a revoked refresh token, all sessions for that user are revoked.
    """
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
        )
    try:
        access_token, new_refresh = await rotate_refresh_token(
            db,
            raw_refresh,
            user_agent=request.headers.get("user-agent"),
        )
    except ValueError as e:
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e) or "Invalid refresh token",
        ) from e

    # Load user for response body
    import jwt as pyjwt
    from app.services.auth_service import decode_token

    payload = decode_token(access_token, expected_type="access")
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalars().first()
    if user is None:
        clear_auth_cookies(response)
        raise HTTPException(status_code=401, detail="User not found")

    await db.commit()
    set_auth_cookies(response, access_token=access_token, refresh_token=new_refresh)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
        "streak": None,
    }


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Revoke refresh token and clear auth cookies."""
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    await revoke_refresh_token(db, raw_refresh)
    await db.commit()
    clear_auth_cookies(response)
    return None


class GoogleAuthRequest(BaseModel):
    credential: str


@router.post("/google", response_model=Token)
async def google_auth(
    body: GoogleAuthRequest,
    response: Response,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate user with Google id_token."""
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google authentication is not configured.")

    try:
        idinfo = id_token.verify_oauth2_token(
            body.credential,
            requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )

        email = idinfo.get("email")
        name = idinfo.get("name", "User")
        google_id = idinfo.get("sub")

        if not email:
            raise HTTPException(status_code=400, detail="Google token does not contain an email address.")

    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    streak_data = None

    if user:
        if not user.google_id:
            user.google_id = google_id
            if user.auth_provider == "local":
                user.auth_provider = "google"
        streak_data = await update_streak_on_login(db, user)
    else:
        user = User(
            email=email,
            username=name,
            hashed_password=None,
            auth_provider="google",
            google_id=google_id,
            is_verified=True,
        )
        db.add(user)
        await db.flush()
        send_welcome_email_task.delay(user.email, user.username)
        streak_data = await update_streak_on_login(db, user)

    access_token, refresh_token = await issue_token_pair(
        db, user.id, user_agent=request.headers.get("user-agent")
    )
    await db.commit()
    await db.refresh(user)

    set_auth_cookies(response, access_token=access_token, refresh_token=refresh_token)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
        "streak": streak_data,
    }
