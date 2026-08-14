"""Shared async Redis helpers for Synapsa.

Provides a small connection pool so quiz cache, metric cache, locks,
and log_broker do not each open one-shot clients on every request.
"""
from __future__ import annotations

import logging
from typing import Optional

from redis import asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Return a process-wide Redis client (lazy singleton)."""
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
    return _pool


async def close_redis() -> None:
    """Shut down the shared client (call from app lifespan shutdown)."""
    global _pool
    if _pool is None:
        return
    try:
        aclose = getattr(_pool, "aclose", None)
        if callable(aclose):
            result = aclose()
            if hasattr(result, "__await__"):
                await result
        else:
            result = _pool.close()
            if hasattr(result, "__await__"):
                await result
    except Exception as e:
        logger.warning(f"[REDIS] close failed: {e}")
    finally:
        _pool = None
