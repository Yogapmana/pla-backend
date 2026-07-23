import asyncio
import uuid
from datetime import datetime
import json

from app.tasks.celery_app import celery_app
from app.db.database import AsyncSession, engine
from app.services.learning_service import LearningService
from app.models.learning import LearningModule as DBLearningModule
from app.utils.log_broker import publish_log
from app.agents.state import (
    SynapsaState, 
    LearningConfig, 
    Curriculum, 
    WeekSchedule, 
    DaySchedule,
    AgentLog,
)
from app.agents.researcher import run_tools_for_query
from app.agents.composer import composer_node
from sqlalchemy import select

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def generate_supplementary_module(self, session_id: str, user_id: str, topic_id: str, supplementary_type: str, context: str = ""):
    """
    Generate supplementary content (remedial or deep_dive) for an existing learning module.
    """
    engine.sync_engine.dispose(close=False)
    
    async def _run():
        async with AsyncSession(engine) as db:
            service = LearningService(db)
            session_uuid = uuid.UUID(session_id)
            session = await service.get_session(session_uuid)
            topic = await service.get_topic(topic_id)
            
            if not session or not topic:
                return {"status": "error", "message": "Session or topic not found"}
            
            existing = await service.get_module(topic_id)
            if not existing:
                return {"status": "error", "message": "Original module not found for supplementary generation"}

            topic_title = topic.title
            existing_id = str(existing.id)

            if supplementary_type == "remedial":
                title_prefix = "Remedial: "
                search_prefix = "Perbaikan pemahaman dasar untuk topik"
            else:
                title_prefix = "Deep Dive: "
                search_prefix = "Studi kasus tingkat lanjut dan pengayaan untuk topik"

            config = LearningConfig(
                topic=session.topic,
                duration_weeks=session.duration_weeks,
                level=session.level,
                hours_per_day=session.hours_per_day,
                language=session.language or "id",
            )
            
            day = DaySchedule(
                day=topic.day_number,
                topic_id=topic_id,
                title=f"{title_prefix}{topic.title}",
                duration_minutes=topic.duration_minutes,
                status="pending",
                search_queries=[f"{search_prefix} {topic.title}"]
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
            
            state: SynapsaState = {
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
                "feedback_actions": [{"action": supplementary_type, "context": context}],
                "agent_logs": [],
            }
            
            queries = day.search_queries
            log = AgentLog(
                timestamp=datetime.utcnow(),
                agent="researcher",
                level="info",
                message=f"Researching supplementary material '{day.title}'..."
            )
            state["agent_logs"].append(log)
            
            all_raw_content = []
            query_tasks = [
                run_tools_for_query(q, topic_id, config.language) for q in queries
            ]
            query_results = await asyncio.gather(*query_tasks)
            
            for results in query_results:
                if isinstance(results, list):
                    all_raw_content.extend(results)
            
            state["research_results"] = all_raw_content
            
            state = composer_node(state)
            
            modules = state.get("modules", [])
            if not modules:
                return {"status": "error", "message": "Composer failed to generate supplementary module"}
            
            generated_module = modules[0]
            
            from sqlalchemy import update
            
            # Update the existing LearningModule using an update statement
            # to avoid MissingGreenlet on expired attributes after long blocking IO
            update_data = {}
            if supplementary_type == "remedial":
                update_data["remedial_markdown"] = generated_module.content_markdown
            elif supplementary_type == "deep_dive":
                update_data["deep_dive_markdown"] = generated_module.content_markdown
                
            if update_data:
                await db.execute(
                    update(DBLearningModule)
                    .where(DBLearningModule.id == existing.id)
                    .values(**update_data)
                )
            
            await db.commit()
            
            await publish_log(session_id, {
                "type": "agent_log",
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "composer",
                "level": "info",
                "message": f"Materi {title_prefix} untuk '{topic_title}' berhasil digenerate.",
                "metadata": {"module_id": str(existing_id), "type": supplementary_type},
            })

            # Create notification
            from app.services.notification_service import NotificationService
            notif_service = NotificationService(db)
            title = "Materi Remedial Tersedia" if supplementary_type == "remedial" else "Materi Deep Dive Tersedia"
            message = f"Materi {title_prefix} untuk topik '{topic_title}' sudah siap Anda pelajari."
            link_path = f"/module/{topic_id}/remedial" if supplementary_type == "remedial" else f"/module/{topic_id}/deep-dive"
            await notif_service.create_notification(
                user_id=user_id,
                title=title,
                message=message,
                notification_type=f"{supplementary_type}_ready",
                link=link_path
            )

            return {
                "status": "success",
                "module_id": existing_id,
                "topic_id": topic_id,
                "type": supplementary_type
            }
    
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(_run())
    except Exception as exc:
        raise self.retry(exc=exc)
