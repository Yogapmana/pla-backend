"""
Test the Redis quiz cache using a mock Redis client.

The cache has three operations: store_quiz, get_quiz, consume_quiz.
The consume path is the most important — it must atomically return the
payload AND delete the key so a quiz can't be submitted twice.
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.quiz_cache import (
    store_quiz,
    get_quiz,
    consume_quiz,
    QUIZ_TTL_SECONDS,
    KEY_PREFIX,
)


def _make_fake_redis():
    """A tiny in-memory redis stub good enough for the cache contract."""
    store = {}

    fake = MagicMock()
    fake.aclose = AsyncMock()
    fake.close = AsyncMock()
    fake.delete = AsyncMock(side_effect=lambda k: store.pop(k, None) is not None)

    async def fake_set(key, value, ex=None):
        store[key] = value

    async def fake_get(key):
        return store.get(key)

    fake.set = AsyncMock(side_effect=fake_set)
    fake.get = AsyncMock(side_effect=fake_get)

    # Pipeline stub for atomic consume
    class FakePipeline:
        def __init__(self):
            self._ops = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, key):
            self._ops.append(("get", key))
            return self

        def delete(self, key):
            self._ops.append(("delete", key))
            return self

        async def execute(self):
            results = []
            for op, key in self._ops:
                if op == "get":
                    results.append(store.get(key))
                elif op == "delete":
                    results.append(store.pop(key, None) is not None)
            return results

    fake.pipeline = MagicMock(return_value=FakePipeline())
    fake._store = store
    return fake


@pytest.mark.asyncio
async def test_store_quiz_returns_quiz_id_and_caches_payload():
    fake = _make_fake_redis()
    with patch("app.services.quiz_cache.aioredis.from_url", return_value=fake):
        questions = [
            {"question": "Q1", "correct_answer": "A", "options": ["A", "B"]},
            {"question": "Q2", "correct_answer": "B", "options": ["A", "B"]},
        ]
        quiz_id = await store_quiz("u1", "t1", questions, ttl_seconds=60)

    assert isinstance(quiz_id, str) and len(quiz_id) >= 8
    # Two keys must be set: the quiz key and the (user, topic) side-index
    set_calls = fake.set.await_args_list
    keys_set = [c.args[0] for c in set_calls]
    assert any(k.startswith(f"{KEY_PREFIX}u1:t1:") and not k.endswith(":current") for k in keys_set)
    assert any(k == f"{KEY_PREFIX}u1:t1:current" for k in keys_set)
    # TTL must be applied
    assert all(c.kwargs.get("ex") == 60 for c in set_calls)


@pytest.mark.asyncio
async def test_get_quiz_returns_payload_when_present():
    fake = _make_fake_redis()
    with patch("app.services.quiz_cache.aioredis.from_url", return_value=fake):
        # Pre-populate as if store_quiz had run
        await store_quiz("u1", "t1", [{"q": 1}], ttl_seconds=60)
        result = await get_quiz("u1", "t1")

    assert result is not None
    assert result["user_id"] == "u1"
    assert result["topic_id"] == "t1"
    assert result["questions"] == [{"q": 1}]
    assert "quiz_id" in result


@pytest.mark.asyncio
async def test_get_quiz_returns_none_on_miss():
    fake = _make_fake_redis()
    with patch("app.services.quiz_cache.aioredis.from_url", return_value=fake):
        result = await get_quiz("nobody", "nothing")

    assert result is None


@pytest.mark.asyncio
async def test_consume_quiz_returns_and_deletes():
    """The whole point of consume: read once, delete after — replay-safe."""
    fake = _make_fake_redis()
    with patch("app.services.quiz_cache.aioredis.from_url", return_value=fake):
        # Populate the cache
        await store_quiz("u1", "t1", [{"q": 1, "correct_answer": "A"}], ttl_seconds=60)
        cached = await get_quiz("u1", "t1")
        quiz_id = cached["quiz_id"]

        # Consume it
        consumed = await consume_quiz("u1", "t1", quiz_id)
        # Second consume must miss
        again = await consume_quiz("u1", "t1", quiz_id)

    assert consumed is not None
    assert consumed["quiz_id"] == quiz_id
    assert consumed["questions"][0]["correct_answer"] == "A"
    assert again is None, "A consumed quiz must not be gradeable again"


@pytest.mark.asyncio
async def test_consume_quiz_clears_side_index_when_match():
    """If the (user, topic) side-index still points to this quiz_id, clear it too."""
    fake = _make_fake_redis()
    with patch("app.services.quiz_cache.aioredis.from_url", return_value=fake):
        await store_quiz("u1", "t1", [{"q": 1}], ttl_seconds=60)
        cached = await get_quiz("u1", "t1")
        await consume_quiz("u1", "t1", cached["quiz_id"])
        # Side-index lookup should now miss
        result = await get_quiz("u1", "t1")

    assert result is None
