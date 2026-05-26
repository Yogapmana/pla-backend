import asyncio
import uuid
from datetime import datetime

from app.tasks.celery_app import celery_app
from app.db.database import AsyncSession, engine
from app.services.learning_service import LearningService
from app.models.learning import LearningModule as DBLearningModule
from app.agents.state import (
    PLAState, 
    LearningConfig, 
    RawContent, 
    AgentLog, 
    Curriculum, 
    WeekSchedule, 
    DaySchedule,
    Message,
    QuizResult,
    ProgressSignals,
    FeedbackAction
)
from app.agents.researcher import run_tools_for_query
from app.agents.composer import composer_node
from app.rag.indexer import index_module

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def generate_module_for_topic(self, session_id: str, user_id: str, topic_id: str):
    """
    Generate a learning module for a specific topic on-demand.
    This runs the researcher → composer → indexer pipeline for a single topic.
    """
    async def _run():
        async with AsyncSession(engine) as db:
            service = LearningService(db)
            
            # 1. Get session and topic info
            session_uuid = uuid.UUID(session_id)
            session = await service.get_session(session_uuid)
            topic = await service.get_topic(topic_id)
            
            if not session or not topic:
                return {"status": "error", "message": "Session or topic not found"}
            
            # 2. Check if module already exists (idempotency)
            existing = await service.get_module(topic_id)
            if existing:
                return {"status": "already_exists", "module_id": str(existing.id)}
            
            # 3. Build minimal state for single topic
            config = LearningConfig(
                topic=session.topic,
                duration_weeks=session.duration_weeks,
                level=session.level,
                hours_per_day=session.hours_per_day,
                language=session.language or "id",
            )
            
            # Build curriculum with just this topic
            day = DaySchedule(
                day=topic.day_number,
                topic_id=topic_id,
                title=topic.title,
                duration_minutes=topic.duration_minutes,
                status="pending",
                search_queries=topic.search_queries or [topic.title],
            )
            week = WeekSchedule(
                week=topic.week_number, 
                title=f"Week {topic.week_number}",
                days=[day]
            )
            curriculum = Curriculum(
                curriculum_id=str(uuid.uuid4()),
                topic=session.topic,
                total_weeks=session.duration_weeks,
                weeks=[week]
            )
            
            state: PLAState = {
                "user_id": user_id,
                "session_id": session_id,
                "learning_config": config,
                "curriculum": curriculum,
                "research_results": [],
                "modules": [],
                "chat_history": [],
                "quiz_results": [],
                "mastery_scores": {},
                "progress_signals": None,
                "feedback_actions": [],
                "agent_logs": [],
            }
            
            # 4. Run researcher
            queries = topic.search_queries or [topic.title]
            log = AgentLog(
                timestamp=datetime.utcnow(),
                agent="researcher",
                level="info",
                message=f"Researching topic '{topic.title}' with {len(queries)} queries..."
            )
            state["agent_logs"].append(log)
            
            all_raw_content = []
            query_tasks = [run_tools_for_query(q, topic_id) for q in queries]
            query_results = await asyncio.gather(*query_tasks)
            
            for results in query_results:
                if isinstance(results, list):
                    all_raw_content.extend(results)
            
            state["research_results"] = all_raw_content
            
            # 5. Run composer
            state = composer_node(state)
            
            # 6. Get generated module
            modules = state.get("modules", [])
            if not modules:
                return {"status": "error", "message": "Composer failed to generate module"}
            
            generated_module = modules[0]
            
            # 7. Save module to DB
            module_rec = DBLearningModule(
                id=uuid.uuid4(),
                topic_id=topic_id,
                session_id=session_uuid,
                title=generated_module.title,
                content_markdown=generated_module.content_markdown,
                sources=generated_module.sources,
            )
            db.add(module_rec)
            await db.flush()
            
            # 8. Index to Qdrant
            index_module(
                user_id=user_id,
                session_id=session_id,
                topic_id=topic_id,
                module_id=str(module_rec.id),
                title=generated_module.title,
                content_markdown=generated_module.content_markdown,
                week=topic.week_number,
                day=topic.day_number,
                sources=generated_module.sources,
            )
            
            # 9. Update topic status to active if still locked
            if topic.status == "locked":
                topic.status = "active"
            
            await db.commit()
            await db.refresh(module_rec)
            
            return {
                "status": "success",
                "module_id": str(module_rec.id),
                "topic_id": topic_id,
                "title": generated_module.title,
            }
    
    try:
        # Use existing event loop if available, or create new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(_run())
    except Exception as exc:
        # Retry on failure
        raise self.retry(exc=exc)
