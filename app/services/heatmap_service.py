"""Heatmap service — aggregate per-day activity for the Streak Heatmap.

The heatmap on the Dashboard renders the last N days (default 84,
i.e. 12 weeks) as a 7-row × N/7-col grid. Each cell shows a score
from 0-3 indicating how active the user was on that day:

  0  : no login that day
  1  : logged in, no other activity
  2  : logged in + 1-2 activities
  3  : logged in + 3+ activities

"Activity" = (modules read) + (quizzes submitted) + (chat messages).
We count each independently, then clamp into the 0-3 scale so the
visual stays clean (the GitHub convention is 4-5 levels, we're
deliberately tighter because the app's "active day" threshold is
low — just opening the app and reading counts as engagement).

We deliberately keep the data flow simple: 4 separate
``GROUP BY DATE()`` queries (one per source), merged in Python.
A single ``UNION ALL`` query would be faster but the Python
merge is easier to read, easier to extend (add new sources), and
fast enough at 84 days of data.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def get_heatmap(
    db: AsyncSession,
    user_id: str,
    days: int = 84,
) -> dict:
    """Build the per-day heatmap payload for a user.

    Returns
    -------
    dict
        ``{"days": int, "start_date": str, "end_date": str,
        "total_logins": int, "total_active_days": int,
        "data": [{date, score, logins, activities}, ...]}``

    The ``data`` list is sorted by date ascending and includes
    EVERY day in the range (even days with score 0) so the
    frontend can render a complete grid without gaps.
    """
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days - 1)
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)

    # ── 1. Logins per day ────────────────────────────────────
    login_rows = await db.execute(
        text("""
            SELECT DATE(logged_in_at AT TIME ZONE 'UTC') AS day,
                   COUNT(*) AS n
            FROM login_events
            WHERE user_id = :user_id
              AND logged_in_at >= :start_dt
            GROUP BY day
        """),
        {"user_id": user_id, "start_dt": start_dt},
    )
    logins_by_day: dict[date, int] = {
        row.day: row.n for row in login_rows
    }

    # ── 2-4. Activity (modules + quizzes + chats) per day ─────
    # We UNION ALL the three sources in a single query, then
    # group by day in Python. This keeps the SQL simple and lets
    # us add new activity sources without a schema migration.
    activity_rows = await db.execute(
        text("""
            SELECT DATE(m.created_at AT TIME ZONE 'UTC') AS day
            FROM learning_modules m
            JOIN learning_sessions s ON s.id = m.session_id
            WHERE s.user_id = :user_id
              AND m.created_at >= :start_dt
            UNION ALL
            SELECT DATE(q.created_at AT TIME ZONE 'UTC') AS day
            FROM quiz_results q
            JOIN learning_sessions s ON s.id = q.session_id
            WHERE s.user_id = :user_id
              AND q.created_at >= :start_dt
            UNION ALL
            SELECT DATE(c.created_at AT TIME ZONE 'UTC') AS day
            FROM chat_messages c
            JOIN learning_sessions s ON s.id = c.session_id
            WHERE s.user_id = :user_id
              AND c.created_at >= :start_dt
        """),
        {"user_id": user_id, "start_dt": start_dt},
    )
    activities_by_day: dict[date, int] = defaultdict(int)
    for row in activity_rows:
        activities_by_day[row.day] += 1

    # ── 5. Build the dense per-day series ───────────────────
    total_logins = 0
    total_active_days = 0
    data: list[dict] = []
    for offset in range(days):
        d = start_date + timedelta(days=offset)
        logins = logins_by_day.get(d, 0)
        activities = activities_by_day.get(d, 0)
        # 0 / 1 / 2-3 / 4+ — see module docstring.
        if logins == 0:
            score = 0
        elif activities == 0:
            score = 1
        elif activities <= 2:
            score = 2
        else:
            score = 3

        if logins > 0:
            total_logins += logins
            total_active_days += 1

        data.append(
            {
                "date": d.isoformat(),
                "score": score,
                "logins": logins,
                "activities": activities,
            }
        )

    return {
        "days": days,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_logins": total_logins,
        "total_active_days": total_active_days,
        "data": data,
    }
