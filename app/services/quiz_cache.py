"""
Redis-backed cache for generated quiz questions.

Why this exists:
- GET /quiz/{topic_id} calls an LLM to generate fresh MCQ + True/False
  questions, which is slow and expensive.
- POST /quiz/submit previously re-generated the quiz on the server side
  if the client didn't send `questions_data` back. That made scores
  unstable (the user could re-take the quiz and get different correct
  answers for the same questions) and wasted tokens.

This cache:
- Stores the generated questions under `quiz:{user_id}:{topic_id}` with a
  30-minute TTL.
- Returns a stable `quiz_id` to the client, which the client must send
  back on submit so we can grade against the exact same questions.
- On submit, the cache entry is consumed (deleted) — a quiz can only be
  submitted once successfully, preventing replay.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from redis import asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

QUIZ_TTL_SECONDS = 30 * 60  # 30 minutes
KEY_PREFIX = "quiz:"


def _key(user_id: str, topic_id: str, quiz_id: str) -> str:
    return f"{KEY_PREFIX}{user_id}:{topic_id}:{quiz_id}"


def _index_key(user_id: str, topic_id: str) -> str:
    """Side-index mapping (user, topic) → current active quiz_id."""
    return f"{KEY_PREFIX}{user_id}:{topic_id}:current"


async def _get_client():
    """Build a one-shot async Redis client (caller closes)."""
    return aioredis.from_url(
        settings.REDIS_URL, encoding="utf-8", decode_responses=True
    )


async def _close(client) -> None:
    """Close redis-py 4.x/5.x compat."""
    try:
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            result = aclose()
            if hasattr(result, "__await__"):
                await result
            return
    except Exception:
        pass
    try:
        result = client.close()
        if hasattr(result, "__await__"):
            await result
    except Exception:
        pass


async def store_quiz(
    user_id: str,
    topic_id: str,
    questions: list[dict[str, Any]],
    ttl_seconds: int = QUIZ_TTL_SECONDS,
) -> str:
    """
    Cache the generated questions and return a new quiz_id.

    The (user, topic) → quiz_id side-index is also updated so that
    lookups by (user, topic) can find the latest active quiz.
    """
    quiz_id = str(uuid.uuid4())
    payload = {
        "quiz_id": quiz_id,
        "user_id": user_id,
        "topic_id": topic_id,
        "questions": questions,
    }
    client = await _get_client()
    try:
        await client.set(
            _key(user_id, topic_id, quiz_id),
            json.dumps(payload, default=str),
            ex=ttl_seconds,
        )
        await client.set(
            _index_key(user_id, topic_id),
            quiz_id,
            ex=ttl_seconds,
        )
        return quiz_id
    except Exception as e:
        # Cache failure is non-fatal — quiz still works, just no replay protection.
        logger.warning(f"[QUIZ_CACHE] store failed for {user_id}/{topic_id}: {e}")
        return quiz_id
    finally:
        await _close(client)


async def get_quiz(
    user_id: str,
    topic_id: str,
    quiz_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Retrieve cached quiz questions. If quiz_id is given, look up that exact
    quiz. If not, fall back to the (user, topic) side-index.

    Returns the full payload (including the list of questions) or None on
    miss / cache error.
    """
    client = await _get_client()
    try:
        if quiz_id is None:
            quiz_id = await client.get(_index_key(user_id, topic_id))
        if not quiz_id:
            return None
        raw = await client.get(_key(user_id, topic_id, quiz_id))
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"[QUIZ_CACHE] get failed for {user_id}/{topic_id}: {e}")
        return None
    finally:
        await _close(client)


async def consume_quiz(
    user_id: str,
    topic_id: str,
    quiz_id: str,
) -> Optional[dict[str, Any]]:
    """
    Atomically read the quiz payload and delete both the quiz key and the
    side-index. The quiz can only be submitted once.

    Returns None if the quiz was missing or already consumed.
    """
    client = await _get_client()
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.get(_key(user_id, topic_id, quiz_id))
            pipe.delete(_key(user_id, topic_id, quiz_id))
            pipe.get(_index_key(user_id, topic_id))
            results = await pipe.execute()
        raw = results[0]
        current = results[2]
        # If the quiz we just consumed is the one in the side-index, clear it too
        if current and current == quiz_id:
            await client.delete(_index_key(user_id, topic_id))
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(
            f"[QUIZ_CACHE] consume failed for {user_id}/{topic_id}/{quiz_id}: {e}"
        )
        return None
    finally:
        await _close(client)
