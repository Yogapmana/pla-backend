import asyncio
import uuid
from datetime import datetime

from app.tasks.celery_app import celery_app
from app.config import settings
from app.agents.orchestrator import pla_graph
from app.agents.state import PLAState, LearningConfig, AgentLog


@celery_app.task(bind=True)
def run_learning_pipeline(self, session_id: str, user_id: str, config: dict):
    """
    Celery task: run full PLA pipeline (planner → researcher → composer).
    Saves curriculum, topics, and modules to DB after completion.
    """
    from app.db.database import async_sessionmaker, AsyncSession, engine
    from app.models.learning import LearningSession, Curriculum, Topic, LearningModule, ResourceLink
    from app.models.agent import AgentLog as DBAgentLog
    from sqlalchemy import select

    async def _run():
        # Clear any cached connection pool tied to a different or closed event loop
        await engine.dispose()

        # Build initial state for LangGraph
        learning_config = LearningConfig(
            topic=config["topic"],
            duration_weeks=config["duration_weeks"],
            level=config["level"],
            hours_per_day=config["hours_per_day"],
            language=config.get("language", "id"),
        )

        initial_state: PLAState = {
            "user_id": user_id,
            "session_id": session_id,
            "learning_config": learning_config,
            "curriculum": None,
            "research_results": [],
            "modules": [],
            "chat_history": [],
            "quiz_results": [],
            "mastery_scores": {},
            "progress_signals": None,
            "feedback_actions": [],
            "agent_logs": [],
        }

        # Execute the LangGraph pipeline
        final_state = await pla_graph.ainvoke(initial_state)

        # Persist results to DB
        async with AsyncSession(engine) as db:
            # Save curriculum
            curriculum = final_state.get("curriculum")
            if curriculum:
                curriculum_rec = Curriculum(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    version=1,
                    curriculum_json=curriculum.model_dump(),
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

            # Save agent logs
            for log in final_state.get("agent_logs", []):
                db_log = DBAgentLog(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    agent=log.agent,
                    level=log.level,
                    message=log.message,
                    metadata_json=log.metadata,
                )
                db.add(db_log)

            # Update session status
            result = await db.execute(
                select(LearningSession).where(LearningSession.id == uuid.UUID(session_id))
            )
            session = result.scalars().first()
            if session:
                session.status = "ready"
                session.completed_at = datetime.utcnow()

            await db.commit()

        # Release all pool connections associated with this event loop before it closes
        await engine.dispose()

        return {
            "session_id": session_id,
            "curriculum_generated": curriculum is not None,
            "modules_count": len(final_state.get("modules", [])),
            "logs_count": len(final_state.get("agent_logs", [])),
        }

    return asyncio.run(_run())