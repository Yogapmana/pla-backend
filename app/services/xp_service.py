"""XP service — level calculation and mastery-milestone XP awards.

XP is the gamification metric that translates the *quality* of
learning (mastery_score) into a single number the user can
internalize ("I'm at level 4 now"). Unlike login streaks (which
are about *consistency*), XP is about *mastery* — you can only
earn it by crossing a real learning milestone.

The signal is the existing ``mastery_score`` on each Topic (0-1,
5-component weighted average). Crossing one of the four
milestones (0.25, 0.5, 0.75, 1.0) awards a fixed XP amount, ONCE
per topic per milestone. We never re-award XP for a milestone
already crossed (enforced by a UNIQUE-style check in
``award_mastery_milestone_xp``).
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.xp_event import XpEvent

logger = logging.getLogger(__name__)


# ── Milestone catalog ─────────────────────────────────────
# Each milestone awards a fixed amount of XP, in increasing
# amounts so the user feels a stronger reward for fully mastering
# a topic. 185 XP per topic is the max. With 24 topics the user
# can theoretically reach ~4,440 XP (Level 5-6).
MASTERY_MILESTONES: list[float] = [0.25, 0.5, 0.75, 1.0]
MILESTONE_XP: dict[float, int] = {
    0.25: 10,    # early engagement — "you're starting to get it"
    0.50: 25,    # halfway — "you've got the basics"
    0.75: 50,    # strong — "you really know this"
    1.00: 100,   # mastered — "topic complete!"
}

# ── Level catalog ─────────────────────────────────────────
# Each level needs ``100 * (2 ** N)`` XP *to advance from* that
# level (so the increment doubles). The "total XP for level N"
# function is therefore ``100 * (2 ** N - 1)``. The first 8
# levels are 0, 100, 300, 700, 1500, 3100, 6300, 12700, 25500.
# Beyond level 10 we just keep the name "Legenda" — the
# mathematical progression doesn't have a hard cap.
LEVEL_NAMES: dict[int, str] = {
    1: "Pemula",
    2: "Penjelajah",
    3: "Pelajar",
    4: "Cendekiawan",
    5: "Ahli",
    6: "Pakar",
    7: "Master",
    8: "Guru",
    9: "Visioner",
    10: "Legenda",
}


def total_xp_for_level(level: int) -> int:
    """Total XP needed to *reach* a given level.

    Level 1 = 0 XP (everyone starts at level 1).
    Level N (N >= 2) = 100 * (2 ** (N - 1) - 1).
    """
    if level <= 1:
        return 0
    return 100 * (2 ** (level - 1) - 1)


def calculate_level(total_xp: int) -> int:
    """Compute the user's current level from their XP total.

    Walks from level 1 upward; returns the highest level whose
    threshold the user has reached. Loops are bounded by a
    ``MAX_LEVEL`` guard so a misconfigured DB row can't send this
    into an infinite loop.
    """
    MAX_LEVEL = 100  # plenty — at level 100 the user has ~10^31 XP
    level = 1
    for candidate in range(2, MAX_LEVEL + 1):
        if total_xp >= total_xp_for_level(candidate):
            level = candidate
        else:
            break
    return level


def xp_in_current_level(total_xp: int) -> dict:
    """Decompose total_xp into current_level / next_level / progress.

    Returns
    -------
    dict
        ``{"level", "level_name", "current_level_xp",
        "next_level_xp", "xp_in_current_level", "xp_to_next_level",
        "progress_pct", "is_max_level"}``
    """
    level = calculate_level(total_xp)
    current_floor = total_xp_for_level(level)
    next_floor = total_xp_for_level(level + 1)
    xp_in_level = total_xp - current_floor
    xp_to_next = next_floor - total_xp
    progress_span = next_floor - current_floor
    progress_pct = (
        round((xp_in_level / progress_span) * 100, 1)
        if progress_span > 0
        else 100
    )
    return {
        "level": level,
        "level_name": LEVEL_NAMES.get(level, LEVEL_NAMES[10]),
        "current_level_xp": current_floor,
        "next_level_xp": next_floor,
        "xp_in_current_level": xp_in_level,
        "xp_to_next_level": xp_to_next,
        "progress_pct": progress_pct,
        # We don't strictly need a max level since the formula
        # works for any N, but the UI uses this flag to swap the
        # "to next level" copy for "max level reached".
        "is_max_level": xp_to_next <= 0 and progress_span <= 0,
    }


def get_crossed_milestones(
    old_mastery: float, new_mastery: float
) -> list[float]:
    """Return the list of milestones crossed going from old to new.

    A milestone is "crossed" iff:
      - ``old_mastery < milestone`` (we hadn't hit it yet)
      - ``new_mastery >= milestone`` (we just hit it)

    Returns a list sorted ascending so the caller can apply
    the XP awards in order. Returns ``[]`` if no milestones
    were crossed (the common case during routine activity).
    """
    return [
        m
        for m in MASTERY_MILESTONES
        if old_mastery < m <= new_mastery
    ]


async def has_already_earned_milestone(
    db: AsyncSession,
    user_id: str,
    topic_id: str,
    milestone: float,
) -> bool:
    """Check if the user has already earned XP for this milestone.

    Looks at the ``xp_events`` audit table for any prior
    ``mastery_milestone`` event with the same topic_id and
    milestone value. The index on (user_id, source, created_at)
    makes this an O(log n) lookup.

    This is the de-dup guarantee. Even if a stale mastery score
    or a recompute causes a milestone to "re-cross", we never
    double-award.
    """
    result = await db.execute(
        select(XpEvent.id)
        .where(XpEvent.user_id == user_id)
        .where(XpEvent.source == "mastery_milestone")
        .where(XpEvent.metadata_json["topic_id"].astext == topic_id)
        .where(XpEvent.metadata_json["milestone"].astext == str(milestone))
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def award_mastery_milestone_xp(
    db: AsyncSession,
    user: User,
    topic_id: str,
    old_mastery: float,
    new_mastery: float,
) -> list[dict]:
    """Award XP for any newly-crossed mastery milestones.

    Parameters
    ----------
    db : AsyncSession
    user : User
        Mutated in place — ``user.total_xp`` is incremented by
        the sum of awarded XP, and ``db.add`` is called for each
        new ``XpEvent`` row. The CALLER is responsible for
        committing the session.
    topic_id : str
        Used as the metadata key for de-dup. The user can't earn
        the same milestone twice for the same topic.
    old_mastery, new_mastery : float
        The mastery score before and after the update. We award
        XP for every milestone where ``old < milestone <= new``.

    Returns
    -------
    list[dict]
        One entry per milestone awarded, shape:
        ``{"milestone": float, "xp": int, "new_total_xp": int,
        "leveled_up": bool, "new_level": int}``

        The list is empty if no milestones were crossed, or if
        every crossed milestone was already earned previously
        (de-dup case). The caller (quiz.py) forwards the list
        to the frontend in the quiz-submit response so the
        user can see "you just earned 25 XP for mastering
        Half of Topic X!".
    """
    crossed = get_crossed_milestones(old_mastery, new_mastery)
    if not crossed:
        return []

    awarded: list[dict] = []
    previous_level = calculate_level(user.total_xp or 0)
    for milestone in crossed:
        # De-dup: a topic can only earn each milestone once.
        if await has_already_earned_milestone(
            db, str(user.id), topic_id, milestone
        ):
            logger.info(
                "[XP] skip dup milestone topic=%s m=%.2f user=%s",
                topic_id, milestone, user.id,
            )
            continue

        amount = MILESTONE_XP[milestone]
        user.total_xp = (user.total_xp or 0) + amount

        event = XpEvent(
            user_id=user.id,
            source="mastery_milestone",
            amount=amount,
            metadata_json={
                "topic_id": topic_id,
                "milestone": milestone,
                "mastery_score": new_mastery,
            },
        )
        db.add(event)

        new_level = calculate_level(user.total_xp)
        awarded.append(
            {
                "milestone": milestone,
                "xp": amount,
                "new_total_xp": user.total_xp,
                "leveled_up": new_level > previous_level,
                "new_level": new_level,
                "level_name": LEVEL_NAMES.get(
                    new_level, LEVEL_NAMES[10]
                ),
            }
        )
        previous_level = new_level
        logger.info(
            "[XP] award topic=%s m=%.2f xp=%d total=%d user=%s",
            topic_id, milestone, amount, user.total_xp, user.id,
        )

    return awarded
