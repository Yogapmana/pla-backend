import asyncio
import uuid
import logging
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)

from app.tasks.celery_app import celery_app
from app.config import settings
from app.agents.orchestrator import build_pla_graph
from app.agents.state import SynapsaState, LearningConfig, AgentLog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool


@celery_app.task(bind=True)
def run_learning_pipeline(self, session_id: str, user_id: str, config: dict):
    """
    Celery task: run full Synapsa pipeline (planner → researcher → composer).
    Saves curriculum, topics, and modules to DB after completion.
    """
    from app.db.database import async_sessionmaker, AsyncSession, engine
    from app.models.learning import LearningSession, Curriculum, Topic, LearningModule, ResourceLink
    from app.models.agent import AgentLog as DBAgentLog
    from sqlalchemy import select, delete as sa_delete
    from app.utils.log_broker import publish_log

    async def _run():
        # Clear any cached connection pool tied to a different or closed event loop.
        # We use sync_engine.dispose(close=False) because the inherited connections
        # from the parent process are tied to a different asyncio loop, and trying
        # to close them asynchronously here will raise a RuntimeError.
        engine.sync_engine.dispose(close=False)

        # ── Idempotency: clean up partial data from previous attempts ──
        # Celery auto-retries on failure. If a previous attempt created
        # topics/modules before crashing, retrying would fail with
        # `duplicate key value violates unique constraint "topics_pkey"`
        # because the LLM-generated topic IDs are deterministic.
        # We delete any partial data for this session before re-inserting.
        #
        # NOTE: Deleting all Curriculum rows for this session also wipes
        # any cached concept_graph_json. The new curriculum row will be
        # inserted with concept_graph_json=None (default), and the
        # ConceptGraphService.version_marker self-healing check will
        # trigger a fresh build on the next read.
        async with AsyncSession(engine) as db:
            try:
                # CASCADE handles modules, quiz_results, progress_signals, etc.
                await db.execute(
                    sa_delete(Topic).where(Topic.session_id == uuid.UUID(session_id))
                )
                await db.execute(
                    sa_delete(Curriculum).where(Curriculum.session_id == uuid.UUID(session_id))
                )
                await db.commit()
            except Exception as cleanup_err:
                logger.warning(f"[CLEANUP] Failed to clean up previous attempt: {cleanup_err}")
                await db.rollback()

        # Build initial state for LangGraph
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

        # Counter log agent yang sudah ditulis ke DB — dipakai agar setiap
        # snapshot dari astream hanya menulis log yang benar-benar baru,
        # bukan re-insert seluruh list (yang akan menggandakan entri).
        persisted_log_count = 0

        # Persist log agent yang baru muncul dari sebuah state snapshot ke DB
        # DAN broadcast via Redis pub/sub agar WebSocket client yang
        # tersambung ke proses manapun (Celery worker / Uvicorn) melihat
        # log real-time, bukan hanya dari proses yang sama.
        async def persist_new_logs(state_snapshot: dict) -> int:
            nonlocal persisted_log_count
            logs = state_snapshot.get("agent_logs") or []
            new_logs = logs[persisted_log_count:]
            if not new_logs:
                return 0

            async with AsyncSession(engine) as db:
                for log in new_logs:
                    db_log = DBAgentLog(
                        id=uuid.uuid4(),
                        session_id=uuid.UUID(session_id),
                        agent=log.agent,
                        level=log.level,
                        message=log.message,
                        metadata_json=log.metadata,
                    )
                    db.add(db_log)
                await db.commit()

            # Broadcast to Redis pub/sub (best-effort, fire-and-forget).
            for log in new_logs:
                await publish_log(session_id, {
                    "type": "agent_log",
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "agent": log.agent,
                    "level": log.level,
                    "message": log.message,
                    "metadata": log.metadata,
                })

            persisted_log_count += len(new_logs)
            return len(new_logs)

        # Execute the LangGraph pipeline dengan streaming
        final_state = initial_state
        db_uri_psycopg = settings.DATABASE_URL.replace("+asyncpg", "")
        
        async with AsyncConnectionPool(db_uri_psycopg, max_size=5, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            
            graph = build_pla_graph().compile(checkpointer=checkpointer)
            config_dict = {"configurable": {"thread_id": session_id}}
            
            async for state_snapshot in graph.astream(initial_state, config=config_dict, stream_mode="values"):
                final_state = state_snapshot
                await persist_new_logs(state_snapshot)

        # Persist results to DB
        async with AsyncSession(engine) as db:
            # Save curriculum
            curriculum = final_state.get("curriculum")
            if curriculum:
                concept_graph = final_state.get("concept_graph")
                if concept_graph:
                    # Sync version marker to initial creation
                    concept_graph["version_marker"] = 1
                
                curriculum_rec = Curriculum(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    version=1,
                    curriculum_json=curriculum.model_dump(),
                    concept_graph_json=concept_graph,
                )
                db.add(curriculum_rec)
                await db.flush()

                # Save topics from curriculum
                for week in curriculum.weeks:
                    for day in week.days:
                        topic_rec = Topic(
                            id=day.topic_id,
                            session_id=uuid.UUID(session_id),
                            curriculum_id=curriculum_rec.id,
                            title=day.title,
                            week_number=week.week,
                            day_number=day.day,
                            duration_minutes=day.duration_minutes,
                            status=day.status,
                            search_queries=day.search_queries,
                        )
                        db.add(topic_rec)
                await db.flush()

            # Save modules
            topic_to_module_id = {}
            for module in final_state.get("modules", []):
                mod_id = uuid.uuid4()
                topic_to_module_id[module.topic_id] = mod_id
                module_rec = LearningModule(
                    id=mod_id,
                    topic_id=module.topic_id,
                    session_id=uuid.UUID(session_id),
                    title=module.title,
                    content_markdown=module.content_markdown,
                    sources=module.sources,
                )
                db.add(module_rec)
            await db.flush()

            # Save resource links (research results)
            for res_content in final_state.get("research_results", []):
                # Map link_type
                link_type = "source"
                if res_content.source_type == "course":
                    link_type = "course"
                elif res_content.source_type == "youtube":
                    link_type = "video"
                elif res_content.source_type in ("arxiv", "semantic_scholar"):
                    link_type = "paper"
                
                # Map platform
                platform = None
                if res_content.source_type == "youtube":
                    platform = "youtube"
                elif res_content.source_type == "arxiv":
                    platform = "arxiv"
                elif res_content.source_type == "semantic_scholar":
                    platform = "semantic_scholar"
                
                # Extract course metadata if course
                price_type = None
                rating = None
                duration = None
                instructor = None
                description = None
                relevant_section = None
                
                if res_content.source_type == "course" and res_content.course_metadata:
                    cm = res_content.course_metadata
                    platform = cm.platform
                    price_type = cm.price_type
                    rating = cm.rating
                    duration = cm.duration
                    instructor = cm.instructor
                    description = cm.description
                    relevant_section = cm.relevant_section
                
                module_id = topic_to_module_id.get(res_content.topic_id)
                
                link_rec = ResourceLink(
                    id=uuid.uuid4(),
                    module_id=module_id,
                    topic_id=res_content.topic_id,
                    session_id=uuid.UUID(session_id),
                    link_type=link_type,
                    title=res_content.source_title,
                    url=res_content.source_url,
                    platform=platform,
                    price_type=price_type,
                    rating=rating,
                    duration=duration,
                    instructor=instructor,
                    description=description,
                    relevant_section=relevant_section,
                    embed_mode="true" if res_content.embed_mode else "false",
                )
                db.add(link_rec)
            await db.flush()

            # Save agent logs sudah dilakukan bertahap oleh persist_new_logs()
            # selama astream — jangan insert ulang di sini atau akan terjadi
            # duplikat entri untuk setiap log.

            # Update session status
            result = await db.execute(
                select(LearningSession).where(LearningSession.id == uuid.UUID(session_id))
            )
            session = result.scalars().first()
            if session:
                # If modules are generated, it means the whole pipeline finished
                if final_state.get("modules"):
                    session.status = "ready"
                    session.completed_at = datetime.utcnow()
                else:
                    # Interrupted after planner
                    session.status = "waiting_approval"

            await db.commit()

        # Release all pool connections associated with this event loop before it closes
        await engine.dispose()

        # NEW (NotebookLM-style mindmap): when the orchestrator
        # successfully produces modules AND marks the session as
        # "ready" (the user is about to be allowed into the
        # dashboard), kick off the background task that scrapes
        # 1-2 sources per topic and builds the 3-level mindmap.
        #
        # Why here, not in `generate_module_for_topic`:
        # The FIRST module is generated synchronously inside the
        # orchestrator's composer_node and saved at the top of
        # this function (line ~187) — ``generate_module_for_topic``
        # is only invoked later, AFTER a quiz submit, for topic 2+.
        # So the trigger that lives in that task never sees the
        # first module. We trigger here instead, exactly at the
        # moment the user is unblocked into the dashboard.
        #
        # Failure is non-fatal: if the broker is down or the task
        # is not registered, the user still has the v1 mindmap
        # (titles-only) as fallback. We log and move on.
        if curriculum is not None and final_state.get("modules"):
            try:
                from app.tasks.generate_enhanced_mindmap import (
                    generate_enhanced_mindmap,
                )
                generate_enhanced_mindmap.delay(session_id)
                logger.info(
                    "[RUN-ORCHESTRATOR] First module saved & session "
                    "ready — queued enhanced mindmap for session %s",
                    session_id,
                )
            except Exception as _trigger_exc:
                logger.warning(
                    "[RUN-ORCHESTRATOR] Failed to queue enhanced "
                    "mindmap task: %s", _trigger_exc,
                )

        return {
            "session_id": session_id,
            "curriculum_generated": curriculum is not None,
            "modules_count": len(final_state.get("modules", [])),
            "logs_count": len(final_state.get("agent_logs", [])),
        }

    try:
        return asyncio.run(_run())
    except Exception as e:
        # Retry on transient errors
        raise self.retry(exc=e)


@celery_app.task(bind=True)
def resume_learning_pipeline(self, session_id: str):
    """
    Celery task: resume the Synapsa pipeline after user approval.
    """
    from app.db.database import async_sessionmaker, AsyncSession, engine
    from app.models.learning import LearningSession, LearningModule, ResourceLink
    from app.models.agent import AgentLog as DBAgentLog
    from sqlalchemy import select
    from app.utils.log_broker import publish_log

    async def _run():
        await engine.dispose()

        persisted_log_count = 0

        async def persist_new_logs(state_snapshot: dict) -> int:
            nonlocal persisted_log_count
            logs = state_snapshot.get("agent_logs") or []
            new_logs = logs[persisted_log_count:]
            if not new_logs:
                return 0

            async with AsyncSession(engine) as db:
                for log in new_logs:
                    db_log = DBAgentLog(
                        id=uuid.uuid4(),
                        session_id=uuid.UUID(session_id),
                        agent=log.agent,
                        level=log.level,
                        message=log.message,
                        metadata_json=log.metadata,
                    )
                    db.add(db_log)
                await db.commit()

            # Broadcast to Redis pub/sub for live WebSocket clients.
            for log in new_logs:
                await publish_log(session_id, {
                    "type": "agent_log",
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "agent": log.agent,
                    "level": log.level,
                    "message": log.message,
                    "metadata": log.metadata,
                })

            persisted_log_count += len(new_logs)
            return len(new_logs)

        final_state = {}
        db_uri_psycopg = settings.DATABASE_URL.replace("+asyncpg", "")
        
        async with AsyncConnectionPool(db_uri_psycopg, max_size=5, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            
            graph = build_pla_graph().compile(checkpointer=checkpointer)
            config_dict = {"configurable": {"thread_id": session_id}}
            
            # Resume graph with empty input since state is already in checkpointer
            async for state_snapshot in graph.astream(None, config=config_dict, stream_mode="values"):
                final_state = state_snapshot
                await persist_new_logs(state_snapshot)

        # Persist remaining results (modules, resource links)
        async with AsyncSession(engine) as db:
            # Re-fetch topic_ids since we don't save topic_to_module_id across tasks easily
            from app.models.learning import Topic
            topic_result = await db.execute(
                select(Topic.id).where(Topic.session_id == uuid.UUID(session_id))
            )
            topic_ids = topic_result.scalars().all()
            # We just need to map them back, or generate new UUIDs
            
            topic_to_module_id = {}
            for module in final_state.get("modules", []):
                mod_id = uuid.uuid4()
                topic_to_module_id[module.topic_id] = mod_id
                module_rec = LearningModule(
                    id=mod_id,
                    topic_id=module.topic_id,
                    session_id=uuid.UUID(session_id),
                    title=module.title,
                    content_markdown=module.content_markdown,
                    sources=module.sources,
                )
                db.add(module_rec)
            await db.flush()

            for res_content in final_state.get("research_results", []):
                link_type = "source"
                if res_content.source_type == "course":
                    link_type = "course"
                elif res_content.source_type == "youtube":
                    link_type = "video"
                elif res_content.source_type in ("arxiv", "semantic_scholar"):
                    link_type = "paper"
                
                platform = None
                if res_content.source_type in ("youtube", "arxiv", "semantic_scholar"):
                    platform = res_content.source_type
                
                price_type = rating = duration = instructor = description = relevant_section = None
                if res_content.source_type == "course" and res_content.course_metadata:
                    cm = res_content.course_metadata
                    platform, price_type, rating, duration, instructor, description, relevant_section = (
                        cm.platform, cm.price_type, cm.rating, cm.duration, cm.instructor, cm.description, cm.relevant_section
                    )
                
                link_rec = ResourceLink(
                    id=uuid.uuid4(),
                    module_id=topic_to_module_id.get(res_content.topic_id),
                    topic_id=res_content.topic_id,
                    session_id=uuid.UUID(session_id),
                    link_type=link_type,
                    title=res_content.source_title,
                    url=res_content.source_url,
                    platform=platform,
                    price_type=price_type,
                    rating=rating,
                    duration=duration,
                    instructor=instructor,
                    description=description,
                    relevant_section=relevant_section,
                    embed_mode="true" if res_content.embed_mode else "false",
                )
                db.add(link_rec)
            await db.flush()

            # Update session status
            result = await db.execute(
                select(LearningSession).where(LearningSession.id == uuid.UUID(session_id))
            )
            session = result.scalars().first()
            if session:
                session.status = "ready"
                session.completed_at = datetime.utcnow()

            await db.commit()

        await engine.dispose()
        return {
            "session_id": session_id,
            "modules_count": len(final_state.get("modules", [])),
            "logs_count": len(final_state.get("agent_logs", [])),
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def run_replan_task(self, session_id: str, action: str, user_id: str):
    """
    Phase polish #4 — Celery task that triggers curriculum replan
    after feedback engine decides repeat/review/accelerate.

    Reuses the same checkpoint-loading + replan_node invocation that
    resume_learning_pipeline uses, but skips the planner/researcher/composer
    phase — it only runs the replan branch.
    """
    from app.db.database import async_sessionmaker, AsyncSession, engine
    from app.models.learning import Curriculum as DBCurriculum, Topic as DBTopic, LearningSession as DBLearningSession
    from app.models.agent import AgentLog as DBAgentLog
    from app.agents.orchestrator import build_pla_graph
    from app.agents.state import SynapsaState, Curriculum, LearningConfig
    from sqlalchemy import select

    async def _run():
        await engine.dispose()

        # Load session curriculum + topics
        async with AsyncSession(engine) as db:
            curriculum = (await db.execute(
                select(DBCurriculum)
                .where(DBCurriculum.session_id == UUID(session_id))
                .order_by(DBCurriculum.version.desc())
                .limit(1)
            )).scalar_one_or_none()
            if not curriculum:
                return {"status": "no_curriculum"}

            latest_topics = (await db.execute(
                select(DBTopic).where(DBTopic.session_id == UUID(session_id))
            )).scalars().all()

            try:
                curriculum_obj = Curriculum.model_validate(curriculum.curriculum_json)
            except Exception as e:
                return {"status": "parse_error", "error": str(e)}

            session_obj = (await db.execute(
                select(DBLearningSession).where(DBLearningSession.id == UUID(session_id))
            )).scalar_one_or_none()
            if not session_obj:
                return {"status": "no_session"}

            learning_config = LearningConfig(
                topic=session_obj.topic,
                duration_weeks=session_obj.duration_weeks,
                level=session_obj.level,
                hours_per_day=session_obj.hours_per_day,
                language=session_obj.language,
                files=session_obj.files
            )

            # Build graph, run only replan branch
            state: SynapsaState = {
                "user_id": "",
                "session_id": session_id,
                "learning_config": learning_config,
                "curriculum": curriculum_obj,
                "research_results": [],
                "modules": [],
                "chat_history": [],
                "quiz_results": [],
                "mastery_scores": {},
                "concept_graph": None,
                "progress_signals": None,
                "feedback_actions": [],  # let graph re-derive from action param
                "agent_logs": [],
            }
            graph = build_pla_graph().compile()
            result = await graph.ainvoke(state)
            new_curr = result.get("curriculum") if isinstance(result, dict) else getattr(result, "curriculum", None)

            if new_curr:
                concept_graph = result.get("concept_graph") if isinstance(result, dict) else getattr(result, "concept_graph", None)
                if concept_graph:
                    concept_graph["version_marker"] = curriculum.version + 1

                new_db_curr = DBCurriculum(
                    id=uuid.uuid4(),
                    session_id=UUID(session_id),
                    version=curriculum.version + 1,
                    curriculum_json=new_curr.model_dump() if hasattr(new_curr, "model_dump") else new_curr,
                    concept_graph_json=concept_graph,
                )
                db.add(new_db_curr)
                
                # Sync Topic rows (Upsert)
                # We upsert topics so new topics are added and existing ones get their status/schedule updated
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                new_topic_id = None
                for week in new_curr.weeks:
                    for day in week.days:
                        # Find the first pending topic to generate a module for
                        if day.status == "pending" and new_topic_id is None:
                            new_topic_id = day.topic_id
                            
                        stmt = pg_insert(DBTopic).values(
                            id=day.topic_id,
                            session_id=UUID(session_id),
                            curriculum_id=new_db_curr.id,
                            title=day.title,
                            week_number=week.week,
                            day_number=day.day,
                            duration_minutes=day.duration_minutes,
                            status=day.status,
                            search_queries=day.search_queries
                        ).on_conflict_do_update(
                            index_elements=['id'],
                            set_={
                                "curriculum_id": new_db_curr.id,
                                "week_number": week.week,
                                "day_number": day.day,
                                "duration_minutes": day.duration_minutes,
                                "status": day.status,
                                "search_queries": day.search_queries
                            }
                        )
                        await db.execute(stmt)

                # Log the auto-replan
                db.add(DBAgentLog(
                    id=uuid.uuid4(),
                    session_id=UUID(session_id),
                    agent="feedback",
                    level="info",
                    message=f"Auto-replan triggered: action={action}",
                ))
                await db.commit()

                # Trigger module generation for the new pending topic
                if new_topic_id:
                    from app.tasks.generate_module import generate_module_for_topic
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"[RESynapsaN] Triggering module generation for adapted topic: {new_topic_id}")
                    generate_module_for_topic.delay(
                        session_id=session_id,
                        user_id=user_id,
                        topic_id=new_topic_id,
                    )

                return {"status": "ok", "new_version": curriculum.version + 1}
            return {"status": "no_change"}

    try:
        return asyncio.run(_run())
    except Exception as e:
        # Retry on transient errors
        raise self.retry(exc=e)

    return asyncio.run(_run())