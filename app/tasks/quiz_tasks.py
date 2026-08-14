"""Celery task for quiz generation — off the request path.

GET /quiz/{topic_id} enqueues this task when no cached quiz exists and returns
202 {status: "generating"}. The task runs the (previously event-loop-blocking)
LLM generation in the worker, keeps the generate lock alive while working,
stores the quiz in Redis, and releases the lock. Clients poll GET /quiz/{topic_id}
until the cached quiz appears (200) or the fail marker is fresh (404).
"""
import asyncio
import logging
import random

from app.tasks.celery_app import celery_app
from app.db.database import engine
from app.agents.tutor import tutor_generate_quiz
from app.services.quiz_cache import (
    store_quiz,
    release_generate_lock,
    refresh_generate_lock,
    mark_quiz_failed,
)

logger = logging.getLogger(__name__)

LOCK_REFRESH_INTERVAL = 60  # seconds — lock TTL is 180s


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def generate_quiz_task(
    self,
    user_id: str,
    topic_id: str,
    topic_title: str,
    language: str = "id",
    num_questions: int = 10,
    difficulty: str = "menengah",
):
    # Clear inherited connections from parent process BEFORE creating the new asyncio loop
    engine.sync_engine.dispose(close=False)

    async def _generate():
        stop = False

        async def _lock_keeper():
            """Refresh the generate lock TTL so waiters keep coalescing on us."""
            while not stop:
                await asyncio.sleep(LOCK_REFRESH_INTERVAL)
                try:
                    await refresh_generate_lock(user_id, topic_id)
                except Exception as e:
                    logger.warning(
                        f"[QUIZ_TASK] lock refresh failed for {user_id}/{topic_id}: {e}"
                    )

        keeper = asyncio.create_task(_lock_keeper())
        try:
            questions = tutor_generate_quiz(
                user_id=user_id,
                topic_id=topic_id,
                topic_title=topic_title,
                language=language,
                num_questions=num_questions,
                difficulty=difficulty,
            )
            if not questions:
                # Both LLM attempts failed — mark so polling doesn't re-enqueue.
                await mark_quiz_failed(user_id, topic_id)
                logger.error(
                    f"[QUIZ_TASK] generation returned empty for {user_id}/{topic_id}"
                )
                return {"status": "failed", "reason": "empty_generation"}
            random.shuffle(questions)
            quiz_id = await store_quiz(
                user_id=user_id, topic_id=topic_id, questions=questions
            )
            return {"status": "done", "quiz_id": quiz_id}
        except Exception as e:
            logger.error(f"[QUIZ_TASK] generation failed for {user_id}/{topic_id}: {e}")
            await mark_quiz_failed(user_id, topic_id)
            raise
        finally:
            stop = True
            if keeper:
                keeper.cancel()
                try:
                    await keeper
                except asyncio.CancelledError:
                    pass
            await release_generate_lock(user_id, topic_id)

    try:
        return asyncio.run(_generate())
    except Exception as exc:
        raise self.retry(exc=exc)