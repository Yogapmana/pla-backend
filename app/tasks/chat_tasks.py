"""Celery tasks for chat: tutor chat, general chatbot, and RAGAS scoring.

Chat is off the request path: POST /chat/message enqueues a task and returns
202 + job_id (the Celery task id). The task runs the (LLM-heavy, previously
event-loop-blocking) agent in the worker, persists the assistant message,
and writes progress to ``chat_job:{job_id}`` in Redis. GET /chat/job/{job_id}
exposes that status to the client.

RAGAS scoring used to be a fire-and-forget asyncio.create_task inside the
request handler; it now runs as its own Celery task so the job status
"done" is not delayed by scoring.
"""
import asyncio
import json
import logging
import uuid

from app.tasks.celery_app import celery_app
from app.db.database import AsyncSession, engine
from app.services.learning_service import LearningService
from app.agents.tutor import tutor_chat
from app.agents.general_chatbot import general_chatbot_chat
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

CHAT_JOB_TTL_SECONDS = 15 * 60  # 15 minutes — client polls must finish within


async def _set_job_status(job_id: str, payload: dict) -> None:
    """Write job status to Redis. Non-fatal on Redis failure."""
    try:
        r = await get_redis()
        await r.set(
            f"chat_job:{job_id}",
            json.dumps(payload, default=str),
            ex=CHAT_JOB_TTL_SECONDS,
        )
    except Exception as e:
        logger.warning(f"[CHAT_TASK] job status write failed for {job_id}: {e}")


def _contexts_from_result(result: dict) -> list[str]:
    """Mirror the old endpoint logic: prefer chunk texts, fall back to titles."""
    chunks = result.get("chunks", [])
    contexts = [c.get("text", "") for c in chunks] if chunks else []
    if not contexts and result.get("sources"):
        contexts = [s.get("title", "") for s in result.get("sources", [])]
    return contexts


async def _persist_and_mark_done(
    session_id: str,
    topic_id: str | None,
    result: dict,
    job_id: str,
    question: str,
) -> None:
    """Save the assistant message, write the done status, queue RAGAS."""
    async with AsyncSession(engine) as db:
        service = LearningService(db)
        saved = await service.save_chat_message(
            session_id=uuid.UUID(session_id),
            topic_id=topic_id,
            role="assistant",
            content=result["response"],
            sources=result.get("sources", []),
            latency_ms=result.get("latency_ms"),
        )
    await _set_job_status(
        job_id,
        {
            "status": "done",
            "message_id": str(saved.id),
            "response": result["response"],
            "sources": result.get("sources", []),
            "latency_ms": result.get("latency_ms"),
            "chunks_used": result.get("chunks_used", 0),
        },
    )
    # Queue RAGAS scoring as its own task (fire-and-forget).
    try:
        ragas_eval_task.delay(
            session_id=session_id,
            message_id=str(saved.id),
            question=question,
            answer=result["response"],
            contexts=_contexts_from_result(result),
        )
    except Exception as e:
        logger.warning(f"[CHAT_TASK] ragas enqueue failed for {job_id}: {e}")


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def tutor_chat_task(
    self,
    user_id: str,
    session_id: str,
    topic_id: str,
    query: str,
    language: str = "id",
    chat_history: list[dict] | None = None,
    include_sources: bool = True,
):
    """RAG tutor chat (topic-based) in the worker; status in Redis."""
    # Clear inherited connections from parent process BEFORE creating the new asyncio loop
    engine.sync_engine.dispose(close=False)
    job_id = self.request.id or str(uuid.uuid4())

    async def _run():
        await _set_job_status(job_id, {"status": "running"})
        try:
            result = tutor_chat(
                user_id=user_id,
                session_id=session_id,
                topic_id=topic_id,
                query=query,
                language=language,
                chat_history=chat_history,
                include_sources=include_sources,
            )
        except Exception as e:
            logger.error(f"[CHAT_TASK] tutor_chat failed for job {job_id}: {e}")
            await _set_job_status(job_id, {"status": "failed", "error": str(e)})
            raise
        await _persist_and_mark_done(
            session_id, topic_id, result, job_id, question=query
        )

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def general_chatbot_task(
    self,
    user_id: str,
    session_id: str,
    query: str,
    chat_history: list[dict] | None = None,
):
    """General chatbot (no topic_id) in the worker; status in Redis."""
    engine.sync_engine.dispose(close=False)
    job_id = self.request.id or str(uuid.uuid4())

    async def _run():
        await _set_job_status(job_id, {"status": "running"})
        try:
            result = await general_chatbot_chat(
                user_id=user_id,
                session_id=session_id,
                query=query,
                chat_history=chat_history,
            )
        except Exception as e:
            logger.error(f"[CHAT_TASK] general_chatbot failed for job {job_id}: {e}")
            await _set_job_status(job_id, {"status": "failed", "error": str(e)})
            raise
        await _persist_and_mark_done(
            session_id, None, result, job_id, question=query
        )

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30)
def ragas_eval_task(
    self,
    session_id: str,
    message_id: str,
    question: str,
    answer: str,
    contexts: list[str],
):
    """Score a chat response with RAGAS outside the request path. Never raises."""
    engine.sync_engine.dispose(close=False)

    async def _run_ragas_background():
        """Background scoring of a chat response — originally in chat.py."""
        from app.rag.evaluator import get_rag_evaluator
        try:
            evaluator = get_rag_evaluator()
            if evaluator is None:
                return
            scores = await evaluator.evaluate(question, answer, contexts)
            if (
                scores.get("rag_faithfulness") is None
                and scores.get("rag_answer_relevancy") is None
            ):
                return
            # Fresh session — background task can't share a request session
            from app.config import settings
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import update
            from app.models.agent import ChatMessage
            engine2 = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
            SessionLocal = async_sessionmaker(engine2, expire_on_commit=False)
            async with SessionLocal() as db:
                await db.execute(
                    update(ChatMessage)
                    .where(ChatMessage.id == uuid.UUID(message_id))
                    .values(
                        rag_faithfulness=scores.get("rag_faithfulness"),
                        rag_answer_relevancy=scores.get("rag_answer_relevancy"),
                    )
                )
                await db.commit()
            if session_id:
                from app.services.metrics_cache import invalidate_session_metrics
                await invalidate_session_metrics(session_id)
            logger.info(
                f"[RAGAS] Scored message {message_id[:8]}... "
                f"faith={scores.get('rag_faithfulness'):.2f} "
                f"rel={scores.get('rag_answer_relevancy'):.2f} "
                f"method={scores.get('method')}"
            )
        except Exception as e:
            logger.error(f"[RAGAS] Background eval failed: {e}")

    try:
        asyncio.run(_run_ragas_background())
    except Exception as e:
        logger.error(f"[RAGAS] task failed: {e}")
