"""Celery task: generate the enhanced (NotebookLM-style) mindmap.

When to run
-----------
Triggered by ``generate_module_for_topic`` after the FIRST module for
a session finishes composing. The user's flow:

  1. Onboarding — planner creates the curriculum (≤ 5s)
  2. Modules 1..N compose in parallel (each ≈ 2-3 min)
  3. As soon as module 1 finishes, ``generate_module_for_topic``
     queues THIS task.
  4. The user can already enter the dashboard (SessionGuard
     transitions ``processing`` → ``ready`` once the first module
     is done). They might navigate to the Curriculum page while
     this task is still running; the frontend shows a "preparing
     mindmap..." state and refreshes when the task finishes.
  5. The user keeps learning. By the time they look at the
     Curriculum page, the enhanced mindmap is usually ready.

What it does
------------
  1. Loads the session's topics (just titles + search_queries).
  2. Calls ``lightweight_researcher`` to scrape 1-2 sources/topic.
  3. Calls ``mindmap_v2_mapper`` to build the 3-level NotebookLM
     mindmap from the scraped text.
  4. Writes the result to ``curriculum.enhanced_mindmap_json``.
  5. Publishes a ``mindmap_enhanced`` event to the WebSocket log
     broker so any open Curriculum page can refetch.

Failure model
-------------
Failures are non-fatal: the user falls back to the v1 mindmap
(stored in ``curriculum.concept_graph_json``) which is always
present from the planner-time generation. We log the error and
publish a ``mindmap_enhanced`` event with status="failed" so the
frontend can hide its spinner.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select

from app.tasks.celery_app import celery_app
from app.db.database import async_sessionmaker, engine
from app.models.learning import Curriculum
from app.services.learning_service import LearningService
from app.services.lightweight_researcher import lightweight_research
from app.agents.mindmap_mapper import mindmap_v2_mapper
from app.utils.log_broker import publish_log

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    name="app.tasks.generate_enhanced_mindmap.generate_enhanced_mindmap",
)
def generate_enhanced_mindmap(self, session_id: str):
    """Build the NotebookLM-style enhanced mindmap for ``session_id``.

    Called by ``generate_module_for_topic`` after the first module
    is composed. Run as a Celery task so it doesn't block the user's
    first dashboard visit.
    """
    return asyncio.run(_run(session_id, self))


async def _run(session_id: str, task) -> dict:
    """Inner async body. Kept separate so the sync Celery wrapper
    can call it via ``asyncio.run`` and so we can catch exceptions
    cleanly and publish a WebSocket failure event.
    """
    from app.config import settings

    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("[ENHANCED-MINDMAP] starting for session %s", session_id)

    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with SessionLocal() as db:
            learning_svc = LearningService(db)

            # 1. Load the latest curriculum for this session.
            curriculum = await learning_svc.get_curriculum(UUID(session_id))
            if not curriculum:
                raise ValueError(
                    f"No curriculum found for session {session_id}"
                )

            cjson = curriculum.curriculum_json or {}
            course_title = (
                cjson.get("title") or cjson.get("topic") or "Kurikulum"
            )
            level = cjson.get("level", "beginner")
            language = cjson.get("language", "id")

            # 2. Load the topics. We need topic_id, title, and
            # search_queries for the lightweight researcher.
            topics = await learning_svc.get_topics(UUID(session_id))
            if not topics:
                raise ValueError(
                    f"No topics found for session {session_id}"
                )

            topic_inputs = [
                {
                    "topic_id": t.id,
                    "title": t.title,
                    "search_queries": t.search_queries or [t.title],
                }
                for t in topics
            ]
            topics_summary = [
                {
                    "topic_id": t.id,
                    "title": t.title,
                    "week_number": t.week_number,
                }
                for t in topics
            ]

            # 3. WebSocket: kick off
            await publish_log(session_id, {
                "type": "mindmap_enhanced",
                "status": "generating",
                "step": "research",
                "progress": 0,
                "timestamp": started_at,
            })

            # 4. Lightweight research — 1-2 sources/topic.
            async def _progress(done, total):
                try:
                    await publish_log(session_id, {
                        "type": "mindmap_enhanced",
                        "status": "generating",
                        "step": "research",
                        "progress": round(done / total, 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass

            research_by_topic = await lightweight_research(
                topic_inputs, progress_cb=_progress
            )

            n_with_sources = sum(
                1 for s in research_by_topic.values() if s
            )
            logger.info(
                "[ENHANCED-MINDMAP] research done — %d/%d topics have sources",
                n_with_sources, len(topic_inputs),
            )

            # 5. WebSocket: mapping
            await publish_log(session_id, {
                "type": "mindmap_enhanced",
                "status": "generating",
                "step": "mapping",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # 6. Build the 3-level mindmap.
            payload = await mindmap_v2_mapper(
                course_title=course_title,
                level=level,
                language=language,
                topics_summary=topics_summary,
                research_by_topic=research_by_topic,
            )
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()

            # 7. Persist. We re-fetch the curriculum row to get a
            # fresh attached instance (the previous one may be
            # detached after the long async work).
            result = await db.execute(
                select(Curriculum).where(
                    Curriculum.session_id == UUID(session_id)
                ).order_by(Curriculum.version.desc()).limit(1)
            )
            current = result.scalar_one_or_none()
            if current is None:
                raise ValueError(
                    f"Curriculum disappeared mid-task for session {session_id}"
                )
            current.enhanced_mindmap_json = payload
            await db.commit()

            stats = payload.get("stats", {})
            logger.info(
                "[ENHANCED-MINDMAP] saved for session %s — %d themes, "
                "%d concepts, %d key points (%.2fs)",
                session_id,
                stats.get("theme_count"),
                stats.get("concept_count"),
                stats.get("key_point_count"),
                payload.get("build_seconds"),
            )

            # 8. WebSocket: done
            await publish_log(session_id, {
                "type": "mindmap_enhanced",
                "status": "ready",
                "stats": stats,
                "timestamp": payload["generated_at"],
            })

            return {
                "status": "success",
                "session_id": session_id,
                "stats": stats,
            }

    except Exception as exc:
        logger.exception(
            "[ENHANCED-MINDMAP] failed for session %s: %s", session_id, exc
        )
        # Publish a failure event so the frontend can hide its
        # "preparing mindmap..." spinner. The user falls back to
        # the v1 mindmap (which was generated at planner time).
        try:
            await publish_log(session_id, {
                "type": "mindmap_enhanced",
                "status": "failed",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass
        # Re-raise so Celery records the failure (and retries up
        # to max_retries). The frontend already got the failure
        # event so the user isn't stuck.
        raise
