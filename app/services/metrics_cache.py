"""Redis cache for expensive aggregated metrics.

Cached endpoints:
  - RAGAS summary per session
  - Daily study time per session
  - User metrics per session
  - Gamification heatmap per user

All failures are non-fatal — callers fall through to the DB path.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

PREFIX = "metrics:"

TTL_RAGAS = 10 * 60          # 10 min
TTL_STUDY_TIME = 5 * 60      # 5 min
TTL_USER_METRICS = 3 * 60    # 3 min
TTL_HEATMAP = 15 * 60        # 15 min


def _ragas_key(session_id: str) -> str:
    return f"{PREFIX}ragas:{session_id}"


def _study_time_key(session_id: str, days: int) -> str:
    return f"{PREFIX}study-time:{session_id}:{days}"


def _user_metrics_key(session_id: str) -> str:
    return f"{PREFIX}user-metrics:{session_id}"


def _heatmap_key(user_id: str, days: int) -> str:
    return f"{PREFIX}heatmap:{user_id}:{days}"


async def cache_get(key: str) -> Optional[Any]:
    try:
        client = await get_redis()
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"[METRICS_CACHE] get failed key={key}: {e}")
        return None


async def cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        client = await get_redis()
        await client.set(key, json.dumps(value, default=str), ex=ttl)
    except Exception as e:
        logger.warning(f"[METRICS_CACHE] set failed key={key}: {e}")


async def cache_delete(*keys: str) -> None:
    if not keys:
        return
    try:
        client = await get_redis()
        await client.delete(*keys)
    except Exception as e:
        logger.warning(f"[METRICS_CACHE] delete failed: {e}")


async def cache_delete_pattern(pattern: str) -> None:
    """Delete keys matching a glob pattern (SCAN, not KEYS)."""
    try:
        client = await get_redis()
        cursor = 0
        while True:
            cursor, batch = await client.scan(cursor=cursor, match=pattern, count=100)
            if batch:
                await client.delete(*batch)
            if cursor == 0:
                break
    except Exception as e:
        logger.warning(f"[METRICS_CACHE] delete_pattern failed pattern={pattern}: {e}")


# ── Typed helpers ────────────────────────────────────────────────────

async def get_ragas_summary(session_id: str) -> Optional[dict]:
    return await cache_get(_ragas_key(str(session_id)))


async def set_ragas_summary(session_id: str, payload: dict) -> None:
    await cache_set(_ragas_key(str(session_id)), payload, TTL_RAGAS)


async def get_daily_study_time(session_id: str, days: int) -> Optional[list]:
    return await cache_get(_study_time_key(str(session_id), days))


async def set_daily_study_time(session_id: str, days: int, payload: list) -> None:
    await cache_set(_study_time_key(str(session_id), days), payload, TTL_STUDY_TIME)


async def get_user_metrics(session_id: str) -> Optional[dict]:
    return await cache_get(_user_metrics_key(str(session_id)))


async def set_user_metrics(session_id: str, payload: dict) -> None:
    await cache_set(_user_metrics_key(str(session_id)), payload, TTL_USER_METRICS)


async def get_heatmap(user_id: str, days: int) -> Optional[dict]:
    return await cache_get(_heatmap_key(str(user_id), days))


async def set_heatmap(user_id: str, days: int, payload: dict) -> None:
    await cache_set(_heatmap_key(str(user_id), days), payload, TTL_HEATMAP)


# ── Invalidation (call from write paths) ─────────────────────────────

async def invalidate_session_metrics(session_id: str) -> None:
    """Drop session-scoped metric caches after quiz/signal/chat writes."""
    sid = str(session_id)
    await cache_delete(
        _ragas_key(sid),
        _user_metrics_key(sid),
    )
    # study-time has variable `days` suffix — wipe by pattern
    await cache_delete_pattern(f"{PREFIX}study-time:{sid}:*")


async def invalidate_user_heatmap(user_id: str) -> None:
    await cache_delete_pattern(f"{PREFIX}heatmap:{user_id}:*")
