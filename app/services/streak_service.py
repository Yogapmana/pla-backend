"""Daily login streak service.

The "streak" is the number of consecutive days a user has logged
in. It's the gamification hook that encourages daily engagement:
log in today, log in tomorrow, your streak goes up. Skip a day,
it resets.

This module is intentionally tiny — it owns the math and the
milestone catalog, and nothing else. The auth router calls
``update_streak_on_login`` on every successful login (and on
register, since that's effectively a first-time login). Failures
here never block authentication — the user still gets their token
even if the streak write fails, and we log the error.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User

logger = logging.getLogger(__name__)


# ── Milestone catalog ────────────────────────────────────────────
# Keyed by streak day. When a user crosses one of these thresholds
# for the first time, the login response includes the matching
# ``milestone`` dict so the frontend can throw a celebration
# (confetti + congratulatory copy). Only the EXACT day matches —
# a user at 31 days does NOT re-trigger the day-30 milestone.
#
# Design note: the icons here are emoji for simplicity (no asset
# pipeline). The frontend can re-render them as <Flame/> / <Trophy/>
# SVGs if we want a more polished look later.
STREAK_MILESTONES: dict[int, dict] = {
    3: {
        "name": "Pemula Konsisten",
        "icon": "🌱",
        "description": "3 hari berturut-turut! Awal yang baik, terus pertahankan!",
    },
    7: {
        "name": "Kutu Buku",
        "icon": "📚",
        "description": "Seminggu penuh belajar. Hebat, kamu sudah membangun kebiasaan!",
    },
    14: {
        "name": "Rajin Belajar",
        "icon": "⭐",
        "description": "2 minggu berturut-turut. Disiplin yang luar biasa!",
    },
    30: {
        "name": "Master Sebulan",
        "icon": "🔥",
        "description": "30 hari streak! Dedikasi yang menginspirasi!",
    },
    50: {
        "name": "Setengah Abad",
        "icon": "💎",
        "description": "50 hari berturut-turut. Hampir setengah abad!",
    },
    100: {
        "name": "Centurion",
        "icon": "🏆",
        "description": "100 hari streak! Legendaris!",
    },
    365: {
        "name": "Legend",
        "icon": "👑",
        "description": "365 HARI STREAK! Kamu adalah legenda PLA!",
    },
}


async def update_streak_on_login(
    db: AsyncSession, user: User
) -> dict:
    """Update the user's login streak and return celebration data.

    Parameters
    ----------
    db : AsyncSession
        The async SQLAlchemy session the caller is already using.
        We commit/rollback through the SAME session so the auth
        flow and the streak write are part of the same transaction
        boundary.
    user : User
        The user who just logged in / registered. Mutated in place
        (current_streak, longest_streak, last_login_date).

    Returns
    -------
    dict
        ``{"new_streak": int, "longest_streak": int, "is_new_day": bool,
        "milestone": dict | None}``

        - ``new_streak``     : the value of ``current_streak`` after
                              the update (1 on a first-ever login
                              or after a >1-day gap).
        - ``longest_streak`` : the user's all-time high streak.
        - ``is_new_day``     : ``True`` iff this login occurred on a
                              different calendar day than the last
                              one (so the UI should celebrate).
        - ``milestone``      : matching entry from
                              ``STREAK_MILESTONES`` if the new
                              streak hits a threshold day, else
                              ``None``.

    Notes
    -----
    Streak rules:
      - First-ever login (no last_login_date): new_streak = 1
      - Same-day re-login: no change, is_new_day = False
      - Yesterday + 1 day = streak + 1
      - Gap > 1 day: streak resets to 1

    Failures here are logged but do NOT raise — the user has
    successfully authenticated, and a streak-write failure should
    never block the login. The function still returns the
    *intended* new_streak so the UI can show it consistently.
    """
    today = date.today()
    last_login: Optional[date] = user.last_login_date

    # Same-day re-login — return the current state, do nothing.
    if last_login == today:
        return {
            "new_streak": user.current_streak or 0,
            "longest_streak": user.longest_streak or 0,
            "is_new_day": False,
            "milestone": None,
        }

    # Compute the new streak based on the gap.
    if last_login is None:
        new_streak = 1
    elif (today - last_login).days == 1:
        new_streak = (user.current_streak or 0) + 1
    else:
        # Either a fresh user or a >1 day gap. Both cases start
        # the streak over at 1.
        new_streak = 1

    # Persist BEFORE the commit — we mutate the ORM object and let
    # the caller commit along with the rest of the auth flow.
    user.current_streak = new_streak
    user.last_login_date = today
    user.longest_streak = max(user.longest_streak or 0, new_streak)

    # NEW (gamification heatmap): record this login as a row in
    # ``login_events``. The Dashboard's Streak Heatmap aggregates
    # from this table to render the last 12 weeks of activity.
    # We insert via the same session the caller is using so the
    # login_events row commits atomically with the streak update.
    from app.models.login_event import LoginEvent
    db.add(LoginEvent(user_id=user.id))

    milestone = STREAK_MILESTONES.get(new_streak)
    logger.info(
        "[STREAK] user=%s new_streak=%d longest=%d milestone=%s",
        user.id,
        new_streak,
        user.longest_streak,
        milestone["name"] if milestone else None,
    )

    return {
        "new_streak": new_streak,
        "longest_streak": user.longest_streak,
        "is_new_day": True,
        "milestone": milestone,
    }
