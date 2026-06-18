"""Gamification router — heatmap + XP/level endpoints.

Mounted in ``app/main.py`` under the ``/api/v1/gamification``
prefix. Both endpoints require auth (``get_current_user``).

The router is intentionally tiny — all the business logic
lives in ``app/services/heatmap_service.py`` and
``app/services/xp_service.py``. The endpoints are thin
serializers over those services.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.xp_event import XpEvent
from app.schemas.gamification import (
    HeatmapResponse,
    XpResponse,
    XpEventSummary,
)
from app.services.heatmap_service import get_heatmap
from app.services.xp_service import (
    LEVEL_NAMES,
    xp_in_current_level,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Human-readable labels for the XP event source discriminator.
# Used by the dashboard's "recent events" feed to render a
# friendly line per event. Keep in sync with the values written
# in ``app/services/xp_service.py``.
SOURCE_LABELS: dict[str, tuple[str, str]] = {
    "mastery_milestone": ("🎯", "Mastery baru"),
}


@router.get("/heatmap", response_model=HeatmapResponse)
async def heatmap_endpoint(
    days: int = Query(
        84,
        ge=14,
        le=365,
        description="Number of days to include (default 84 = 12 weeks). "
        "Min 14 (2 weeks), max 365 (1 year).",
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Per-day login + activity intensity for the Streak Heatmap.

    See ``app/services/heatmap_service.get_heatmap`` for the data
    shape. The frontend renders this as a 7×N grid (days of week
    × weeks).
    """
    payload = await get_heatmap(db, str(current_user.id), days=days)
    return payload


@router.get("/xp", response_model=XpResponse)
async def xp_endpoint(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Current XP total, level info, and recent XP events.

    The level is computed live from ``User.total_xp`` via
    ``xp_in_current_level`` — the same formula the dashboard
    uses to render the progress bar.

    ``recent_events`` returns the last 5 XpEvent rows for the
    activity feed. The full audit log is in ``xp_events`` if
    we ever need a paginated history view.
    """
    level_info = xp_in_current_level(current_user.total_xp or 0)

    # Pull the most recent 5 events, newest first.
    result = await db.execute(
        select(XpEvent)
        .where(XpEvent.user_id == current_user.id)
        .order_by(XpEvent.created_at.desc())
        .limit(5)
    )
    events = result.scalars().all()

    recent: list[XpEventSummary] = []
    for e in events:
        icon, base_label = SOURCE_LABELS.get(e.source, ("⭐", "XP"))
        # Compose a human-friendly label. For ``mastery_milestone``
        # events we have the topic + milestone in metadata, so we
        # can render something specific like "Mastery 50% • Topic X".
        label = base_label
        if e.source == "mastery_milestone" and e.metadata_json:
            topic_id = e.metadata_json.get("topic_id")
            milestone = e.metadata_json.get("milestone")
            if milestone is not None:
                ms = int(round(float(milestone) * 100))
                label = f"Mastery {ms}%"
                if topic_id:
                    label = f"{label} • Topik"

        recent.append(
            XpEventSummary(
                id=str(e.id),
                source=e.source,
                amount=e.amount,
                label=label,
                icon=icon,
                created_at=(
                    e.created_at.isoformat() if e.created_at else ""
                ),
                metadata=e.metadata_json,
            )
        )

    return XpResponse(
        total_xp=current_user.total_xp or 0,
        level_info=level_info,
        recent_events=recent,
    )
