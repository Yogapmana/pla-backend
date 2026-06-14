"""
Test the Redis log broker using a mock Redis client.

We don't need a real Redis instance to verify channel naming, payload
serialization, and error handling — all of which are the only logic
in log_broker.py outside the redis-py calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.utils.log_broker import channel_for, publish_log, subscribe_to_session


def test_channel_for():
    """Channel name follows the pla:agent-logs:{session_id} convention."""
    assert channel_for("abc-123") == "pla:agent-logs:abc-123"
    assert channel_for("00000000-0000-0000-0000-000000000000") == \
        "pla:agent-logs:00000000-0000-0000-0000-000000000000"


@pytest.mark.asyncio
async def test_publish_log_serializes_payload():
    """publish_log should JSON-encode the payload and call redis.publish."""
    fake_redis = MagicMock()
    fake_redis.publish = AsyncMock(return_value=3)
    fake_redis.aclose = AsyncMock()
    fake_redis.close = AsyncMock()

    with patch("app.utils.log_broker.aioredis.from_url", return_value=fake_redis):
        count = await publish_log(
            "sess-1",
            {
                "type": "agent_log",
                "agent": "planner",
                "level": "info",
                "message": "hi",
                "metadata": {"k": "v"},
            },
        )

    assert count == 3
    fake_redis.publish.assert_awaited_once()
    chan, body = fake_redis.publish.await_args.args
    assert chan == "pla:agent-logs:sess-1"
    # Body should be a JSON string containing all fields
    import json
    decoded = json.loads(body)
    assert decoded["agent"] == "planner"
    assert decoded["message"] == "hi"
    assert decoded["metadata"] == {"k": "v"}


@pytest.mark.asyncio
async def test_publish_log_swallows_errors():
    """If Redis is unreachable, publish_log returns 0 instead of raising."""
    fake_redis = MagicMock()
    fake_redis.publish = AsyncMock(side_effect=ConnectionError("redis down"))
    fake_redis.aclose = AsyncMock()
    fake_redis.close = AsyncMock()

    with patch("app.utils.log_broker.aioredis.from_url", return_value=fake_redis):
        count = await publish_log("sess-1", {"type": "agent_log", "message": "x"})

    assert count == 0  # graceful — pub/sub is best-effort


@pytest.mark.asyncio
async def test_subscribe_to_session_invokes_handler():
    """subscribe_to_session should call on_message for every published payload."""
    received = []

    async def handler(payload):
        received.append(payload)

    # Build a fake pubsub object that yields a couple of messages then ends
    fake_pubsub = MagicMock()
    fake_pubsub.subscribe = AsyncMock()
    fake_pubsub.unsubscribe = AsyncMock()
    fake_pubsub.aclose = AsyncMock()
    fake_pubsub.close = AsyncMock()

    async def fake_listen():
        yield {"type": "subscribe", "channel": "pla:agent-logs:s1", "data": 1}
        yield {"type": "message", "channel": "pla:agent-logs:s1",
               "data": '{"type":"agent_log","agent":"planner","message":"hello"}'}
        yield {"type": "message", "channel": "pla:agent-logs:s1",
               "data": '{"type":"agent_log","agent":"composer","message":"done"}'}
        # Stop iteration: the test will end the loop by cancelling

    fake_pubsub.listen = fake_listen

    fake_redis = MagicMock()
    fake_redis.aclose = AsyncMock()
    fake_redis.close = AsyncMock()

    with patch("app.utils.log_broker.aioredis.from_url", return_value=fake_redis):
        with patch("app.utils.log_broker.aioredis.client.PubSub", return_value=fake_pubsub):
            # Make pubsub() return our fake (it lives on the client instance)
            fake_redis.pubsub = MagicMock(return_value=fake_pubsub)

            task = asyncio.create_task(subscribe_to_session("s1", handler))
            # Let it consume the two messages we yielded
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    assert len(received) == 2
    assert received[0]["agent"] == "planner"
    assert received[1]["agent"] == "composer"


import asyncio  # at bottom to keep import grouping tidy above
