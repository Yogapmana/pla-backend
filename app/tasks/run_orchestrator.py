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
    from app.models.learning import LearningSession, Curriculum, Topic, LearningModule
    from app.models.agent import AgentLog as DBAgentLog
    from sqlalchemy import select

    async def _run():
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

            # Save modules
            for module in final_state.get("modules", []):
                module_rec = LearningModule(
                    id=uuid.uuid4(),
                    topic_id=module.topic_id,
                    session_id=uuid.UUID(session_id),
                    title=module.title,
                    content_markdown=module.content_markdown,
                    sources=module.sources,
                )
                db.add(module_rec)

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

        return {
            "session_id": session_id,
            "curriculum_generated": curriculum is not None,
            "modules_count": len(final_state.get("modules", [])),
            "logs_count": len(final_state.get("agent_logs", [])),
        }

    return asyncio.run(_run())