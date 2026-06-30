from datetime import date
from typing import Optional
from pydantic import BaseModel, EmailStr
from uuid import UUID


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    language_preference: str = "id"


class UserLanguageUpdate(BaseModel):
    language_preference: str


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: EmailStr
    language_preference: str
    # Gamification streak fields — read from the users table.
    # `current_streak` and `longest_streak` default to 0 for
    # users who haven't logged in yet; `last_login_date` is None
    # until the first login populates it.
    current_streak: int = 0
    longest_streak: int = 0
    last_login_date: Optional[date] = None

    class Config:
        from_attributes = True


class StreakInfo(BaseModel):
    """Streak update metadata returned on login/register.

    - ``new_streak``     : value of current_streak after this login
    - ``longest_streak`` : user's all-time high streak
    - ``is_new_day``     : True iff this login was on a different
                            calendar day than the previous one (so
                            the UI should trigger a celebration)
    - ``milestone``      : milestone dict (name, icon, description)
                            from streak_service.STREAK_MILESTONES,
                            or None if the new streak isn't a
                            threshold day.
    """

    new_streak: int
    longest_streak: int
    is_new_day: bool
    milestone: Optional[dict] = None


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
    # Streak info is included on login/register. On /me and
    # /refresh, the response doesn't include this field (it stays
    # null) — the user has already been through login once.
    streak: Optional[StreakInfo] = None


class TokenData(BaseModel):
    user_id: str | None = None
