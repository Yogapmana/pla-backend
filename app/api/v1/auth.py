from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.database import get_db
from app.schemas.auth import UserCreate, UserResponse, Token
from app.models.user import User
from app.services.auth_service import get_password_hash, verify_password, create_access_token
from app.services.streak_service import update_streak_on_login
from app.dependencies import get_current_user
from app.config import settings

router = APIRouter()


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_in.email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user_in.password)
    new_user = User(
        email=user_in.email,
        username=user_in.username,
        hashed_password=hashed_password,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # NOTE: We intentionally do NOT call ``update_streak_on_login``
    # here. Register creates the user with ``last_login_date=NULL``,
    # so the user's first real login (the one right after
    # Register) will see ``last_login is None`` and set streak=1
    # with ``is_new_day=True`` — which triggers the welcome
    # celebration modal. If we initialized the streak on
    # Register, the very next login would be a same-day re-login
    # (``is_new_day=False``) and the user would miss the "you
    # started a streak!" moment.
    #
    # If we ever change the frontend to auto-login after Register
    # (i.e. skip the /login step), we'll need to revisit this and
    # re-enable the streak init here, plus handle the "celebration
    # happens on the register response, not the login response"
    # path in the auth store.

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(new_user.id)}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": new_user,
        # streak is None on Register — see the comment block above
        # for why we don't initialize the streak here. The first
        # login will populate it and queue the celebration.
        "streak": None,
    }


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalars().first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update the user's daily login streak. The service mutates
    # the user in place and returns metadata (new_streak,
    # is_new_day, milestone) that the frontend uses to decide
    # whether to show a celebration modal.
    streak_data = await update_streak_on_login(db, user)
    # Commit the streak update alongside the auth flow so they
    # share the same transaction boundary.
    await db.commit()
    await db.refresh(user)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
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


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Refresh access token for authenticated user."""
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(current_user.id)}, expires_delta=access_token_expires
    )
    # No streak update on refresh — same reason as /me above.
    return {"access_token": access_token, "token_type": "bearer", "user": current_user, "streak": None}
