"""Gamification API schemas — heatmap, XP, level info."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Heatmap ─────────────────────────────────────────────
class HeatmapDay(BaseModel):
    """One cell in the heatmap grid.

    - ``date``     : ISO 8601 date string (YYYY-MM-DD)
    - ``score``    : 0-3 intensity level (0=empty, 1=login only,
                     2=login+some activity, 3=login+lots)
    - ``logins``   : number of login_events on that date
    - ``activities`` : total activity events (modules + quizzes +
                       chat messages) on that date
    """
    date: str
    score: int
    logins: int
    activities: int


class HeatmapResponse(BaseModel):
    days: int
    start_date: str
    end_date: str
    total_logins: int
    total_active_days: int
    data: list[HeatmapDay]


# ── XP / Level ───────────────────────────────────────────
class XpLevelInfo(BaseModel):
    level: int
    level_name: str
    current_level_xp: int
    next_level_xp: int
    xp_in_current_level: int
    xp_to_next_level: int
    progress_pct: float
    is_max_level: bool


class XpEventSummary(BaseModel):
    """One row of the XP history feed."""
    id: str
    source: str
    amount: int
    label: str
    icon: str
    created_at: str
    metadata: Optional[dict] = None


class XpResponse(BaseModel):
    total_xp: int
    level_info: XpLevelInfo
    recent_events: list[XpEventSummary] = Field(default_factory=list)
