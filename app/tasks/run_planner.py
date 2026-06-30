import asyncio
import uuid
from celery import shared_task
from app.tasks.celery_app import celery_app


@celery_app.task(bind=True)
def run_planner_task(self, session_id: str, config: dict):
    """
    Celery task: run only the Planner Agent to generate/revise curriculum.
    Useful for:
    - Initial curriculum generation
    - Revising schedule after feedback (adaptive loop)
    """
    from app.db.database import async_sessionmaker, AsyncSession, engine
    from app.models.learning import Curriculum
    from sqlalchemy import select

    async def _run():
        await engine.dispose()

        from app.agents.planner import PlannerAgent
        from app.agents.state import SynapsaState, LearningConfig

        planner = PlannerAgent()
        learning_config = LearningConfig(
            topic=config["topic"],
            duration_weeks=config["duration_weeks"],
            level=config["level"],
            hours_per_day=config["hours_per_day"],
            language=config.get("language", "id"),
        )

        curriculum = await planner.generate_curriculum(learning_config)

        async with AsyncSession(engine) as db:
            curriculum_rec = Curriculum(
                id=uuid.uuid4(),
                session_id=uuid.UUID(session_id),
                version=config.get("version", 1),
                curriculum_json=curriculum.model_dump(),
            )
            db.add(curriculum_rec)
            await db.commit()
            await db.refresh(curriculum_rec)

        await engine.dispose()
        return {
            "session_id": session_id,
            "curriculum_id": str(curriculum_rec.id),
            "weeks_count": len(curriculum.weeks),
        }

    return asyncio.run(_run())


@shared_task
def run_planner_async(session_id: str, config: dict):
    """Async wrapper for run_planner_task (can be called from other tasks)."""
    return run_planner_task.delay(session_id, config)