"""
Redis-backed pub/sub broker for real-time agent log broadcasting.

Why this exists:
- Agent log events are produced inside Celery worker processes
  (run_learning_pipeline / generate_module_for_topic).
- WebSocket clients connect to Uvicorn worker processes.
- Without pub/sub, in-process ConnectionManager cannot deliver logs
  from one process to another — clients only see logs when they happen
  in the same process that handles their socket.

The broker publishes JSON-serialised log dicts on a per-session channel:

    pla:agent-logs:{session_id}

WebSocket handlers subscribe to that channel and forward each message
to their in-process ConnectionManager.broadcast().
"""
from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Optional, Union

from redis import asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "pla:agent-logs:"


def channel_for(session_id: str) -> str:
    """Redis channel name for a given session id."""
    return f"{CHANNEL_PREFIX}{session_id}"


async def _close_client(client) -> None:
    """Close a redis-py client in a version-compatible way (4.x vs 5.x)."""
    try:
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            result = aclose()
            if hasattr(result, "__await__"):
                await result
            return
    except Exception as e:
        logger.debug(f"[LOG_BROKER] aclose failed (non-fatal): {e}")
    try:
        result = client.close()
        if hasattr(result, "__await__"):
            await result
    except Exception as e:
        logger.debug(f"[LOG_BROKER] close failed (non-fatal): {e}")


async def publish_log(session_id: str, log_payload: dict[str, Any]) -> int:
    """
    Publish one agent log entry to the session's Redis channel.

    Returns the number of subscribers that received the message.
    Returns 0 on connection error (caller can ignore — pub/sub is best-effort).
    """
    client = aioredis.from_url(
        settings.REDIS_URL, encoding="utf-8", decode_responses=True
    )
    try:
        count = await client.publish(
            channel_for(session_id), json.dumps(log_payload, default=str)
        )
        return int(count or 0)
    except Exception as e:
        logger.warning(f"[LOG_BROKER] publish failed for session {session_id}: {e}")
        return 0
    finally:
        await _close_client(client)


OnMessage = Callable[[dict[str, Any]], Union[None, Awaitable[None]]]


async def subscribe_to_session(
    session_id: str,
    on_message: OnMessage,
    *,
    on_connect: Optional[Callable[[], Union[None, Awaitable[None]]]] = None,
) -> None:
    """
    Long-running coroutine: subscribe to a session's log channel and call
    on_message(parsed_dict) for every received payload.

    Should be spawned as a background asyncio.Task from a WebSocket handler.
    on_connect is invoked once the subscription is established (optional).
    """
    client = aioredis.from_url(
        settings.REDIS_URL, encoding="utf-8", decode_responses=True
    )
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(channel_for(session_id))
        if on_connect is not None:
            result = on_connect()
            if hasattr(result, "__await__"):
                await result
        async for message in pubsub.listen():
            if message is None:
                continue
            if message.get("type") != "message":
                continue  # ignore subscribe/unsubscribe confirmations
            raw = message.get("data")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning(
                    f"[LOG_BROKER] non-JSON payload on {session_id}: {raw!r}"
                )
                continue
            try:
                result = on_message(payload)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.exception(f"[LOG_BROKER] on_message handler failed: {e}")
    finally:
        try:
            await pubsub.unsubscribe(channel_for(session_id))
        except Exception:
            pass
        await _close_client(pubsub)
        await _close_client(client)
