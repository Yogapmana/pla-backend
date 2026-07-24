import asyncio
import logging
import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.agents.orchestrator import build_pla_graph
from app.agents.state import LearningConfig, SynapsaState
from app.config import settings
from app.tasks.celery_app import celery_app
from app.tasks.pipeline_persist import (
    mark_session_status,
    persist_new_logs,
    save_curriculum_and_topics,
    save_modules_and_links,
)

logger = logging.getLogger(__name__)


async def _mark_session_failed(session_id: str, error: str) -> None:
    """Set learning session status to failed and broadcast an error log.

    Called only after Celery retries are exhausted so AgentLoadingScreen
    (and any other poller) can leave the processing spinner.
    """
    from sqlalchemy import select

    from app.db.database import AsyncSession, engine
    from app.models.agent import AgentLog as DBAgentLog
    from app.models.learning import LearningSession
    from app.utils.log_broker import publish_log

    engine.sync_engine.dispose(close=False)
    try:
        async with AsyncSession(engine) as db:
            result = await db.execute(
                select(LearningSession).where(LearningSession.id == uuid.UUID(session_id))
            )
            session = result.scalars().first()
            if session is not None:
                session.status = "failed"
            db.add(
                DBAgentLog(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    agent="orchestrator",
                    level="error",
                    message=f"Pipeline gagal: {error[:500]}",
                    metadata_json={"error": error[:1000]},
                )
            )
            await db.commit()
        await publish_log(
            session_id,
            {
                "type": "agent_log",
                "agent": "orchestrator",
                "level": "error",
                "message": f"Pipeline gagal: {error[:300]}",
            },
        )
    except Exception as mark_err:
        logger.exception(
            "[RUN-ORCHESTRATOR] failed to mark session %s as failed: %s",
            session_id,
            mark_err,
        )
    finally:
        try:
            await engine.dispose()
        except Exception:
            pass


def _handle_pipeline_failure(task, session_id: str, exc: Exception):
    """Retry transient failures; on final attempt mark session failed."""
    logger.exception("[RUN-ORCHESTRATOR] pipeline failed session=%s: %s", session_id, exc)
    max_retries = task.max_retries if task.max_retries is not None else 0
    if task.request.retries >= max_retries:
        try:
            asyncio.run(_mark_session_failed(session_id, str(exc)))
        except Exception as mark_exc:
            logger.exception(
                "[RUN-ORCHESTRATOR] mark-failed crashed for %s: %s",
                session_id,
                mark_exc,
            )
        return {"status": "failed", "session_id": session_id, "error": str(exc)}
    raise task.retry(exc=exc)


async def _stream_graph(session_id: str, initial_state: dict | None) -> dict:
    """Run compiled graph with checkpointer; stream agent logs to DB/Redis."""
    final_state = initial_state or {}
    persisted_log_count = 0
    db_uri_psycopg = settings.DATABASE_URL.replace("+asyncpg", "")

    async with AsyncConnectionPool(
        db_uri_psycopg, max_size=5, kwargs={"autocommit": True}
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        graph = build_pla_graph().compile(checkpointer=checkpointer)
        config_dict = {"configurable": {"thread_id": session_id}}

        async for state_snapshot in graph.astream(
            initial_state, config=config_dict, stream_mode="values"
        ):
            final_state = state_snapshot
            persisted_log_count = await persist_new_logs(
                session_id, state_snapshot, persisted_log_count
            )

    return final_state


async def _cleanup_partial_session(session_id: str) -> None:
    """Delete partial curriculum/topics before a retry (idempotent re-run)."""
    from sqlalchemy import delete as sa_delete

    from app.db.database import AsyncSession, engine
    from app.models.learning import Curriculum, Topic

    async with AsyncSession(engine) as db:
        try:
            await db.execute(
                sa_delete(Topic).where(Topic.session_id == uuid.UUID(session_id))
            )
            await db.execute(
                sa_delete(Curriculum).where(Curriculum.session_id == uuid.UUID(session_id))
            )
            await db.commit()
        except Exception as cleanup_err:
            logger.warning("[CLEANUP] Failed to clean up previous attempt: %s", cleanup_err)
            await db.rollback()


async def _notify_curriculum_ready(user_id: str, session_id: str) -> None:
    from app.db.database import AsyncSession, engine
    from app.services.notification_service import NotificationService

    try:
        async with AsyncSession(engine) as notif_db:
            notif_service = NotificationService(notif_db)
            await notif_service.create_notification(
                user_id=user_id,
                title="Kurikulum Berhasil Disusun",
                message=(
                    "Kurikulum belajar Anda telah berhasil disusun. "
                    "Mari mulai belajar hari ini!"
                ),
                notification_type="curriculum_ready",
                link=f"/dashboard/{session_id}",
            )
    except Exception as notif_exc:
        logger.warning(
            "[RUN-ORCHESTRATOR] Failed to send curriculum notification: %s", notif_exc
        )


async def _queue_enhanced_mindmap(session_id: str) -> None:
    try:
        from app.tasks.generate_enhanced_mindmap import generate_enhanced_mindmap

        generate_enhanced_mindmap.delay(session_id)
        logger.info(
            "[RUN-ORCHESTRATOR] queued enhanced mindmap for session %s", session_id
        )
    except Exception as _trigger_exc:
        logger.warning(
            "[RUN-ORCHESTRATOR] Failed to queue enhanced mindmap task: %s",
            _trigger_exc,
        )


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def run_learning_pipeline(self, session_id: str, user_id: str, config: dict):
    """
    Celery task: run full Synapsa pipeline (planner → researcher → composer).
    Saves curriculum, topics, and modules to DB after completion.
    """
    from app.db.database import AsyncSession, engine

    async def _run():
        engine.sync_engine.dispose(close=False)
        await _cleanup_partial_session(session_id)

        learning_config = LearningConfig(
            topic=config["topic"],
            duration_weeks=config["duration_weeks"],
            level=config["level"],
            hours_per_day=config["hours_per_day"],
            language=config.get("language", "id"),
        )

        initial_state: SynapsaState = {
            "user_id": user_id,
            "session_id": session_id,
            "learning_config": learning_config,
            "curriculum": None,
            "research_results": [],
            "modules": [],
            "chat_history": [],
            "quiz_results": [],
            "mastery_scores": {},
            "concept_graph": None,
            "progress_signals": None,
            "feedback_actions": [],
            "agent_logs": [],
        }

        final_state = await _stream_graph(session_id, initial_state)
        curriculum = final_state.get("curriculum")
        modules = final_state.get("modules", [])

        async with AsyncSession(engine) as db:
            await save_curriculum_and_topics(
                db,
                session_id,
                curriculum,
                final_state.get("concept_graph"),
            )
            topic_module_map = await save_modules_and_links(
                db,
                session_id,
                modules,
                final_state.get("research_results", []),
            )
            await mark_session_status(db, session_id, ready=bool(modules))
            await db.commit()

        await engine.dispose()

        # Run embedding indexer for each module
        from app.rag.indexer import index_module
        for module in modules:
            module_id = str(topic_module_map.get(module.topic_id, ""))
            await asyncio.to_thread(
                index_module,
                user_id=user_id,
                session_id=session_id,
                topic_id=module.topic_id,
                module_id=module_id,
                title=module.title,
                content_markdown=module.content_markdown,
            )

        if curriculum is not None and modules:
            await _queue_enhanced_mindmap(session_id)
            await _notify_curriculum_ready(user_id, session_id)

        return {
            "session_id": session_id,
            "curriculum_generated": curriculum is not None,
            "modules_count": len(modules),
            "logs_count": len(final_state.get("agent_logs", [])),
        }

    try:
        return asyncio.run(_run())
    except Exception as e:
        return _handle_pipeline_failure(self, session_id, e)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def resume_learning_pipeline(self, session_id: str):
    """Celery task: resume the Synapsa pipeline after user approval."""
    from app.db.database import AsyncSession, engine

    async def _run():
        await engine.dispose()

        final_state = await _stream_graph(session_id, None)
        modules = final_state.get("modules", [])

        from app.models.learning import LearningSession
        from sqlalchemy import select
        async with AsyncSession(engine) as db:
            topic_module_map = await save_modules_and_links(
                db,
                session_id,
                modules,
                final_state.get("research_results", []),
            )
            await mark_session_status(db, session_id, ready=True)
            result = await db.execute(
                select(LearningSession).where(LearningSession.id == uuid.UUID(session_id))
            )
            session = result.scalars().first()
            user_id = str(session.user_id) if session else ""
            await db.commit()

        await engine.dispose()

        # Run embedding indexer for each module
        from app.rag.indexer import index_module
        for module in modules:
            module_id = str(topic_module_map.get(module.topic_id, ""))
            await asyncio.to_thread(
                index_module,
                user_id=user_id,
                session_id=session_id,
                topic_id=module.topic_id,
                module_id=module_id,
                title=module.title,
                content_markdown=module.content_markdown,
            )

        return {
            "session_id": session_id,
            "modules_count": len(modules),
            "logs_count": len(final_state.get("agent_logs", [])),
        }

    try:
        return asyncio.run(_run())
    except Exception as e:
        return _handle_pipeline_failure(self, session_id, e)
