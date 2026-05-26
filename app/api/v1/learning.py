from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.learning import LearningSession, Topic, LearningModule
from app.services.learning_service import LearningService
from app.tasks.run_orchestrator import run_learning_pipeline
from app.tasks.generate_module import generate_module_for_topic

router = APIRouter()


class LearningStartRequest(BaseModel):
    topic: str
    duration_weeks: int
    level: str
    hours_per_day: float
    language: str = "id"


class LearningSessionResponse(BaseModel):
    id: str
    topic: str
    level: str
    duration_weeks: int
    hours_per_day: float
    language: str
    status: str
    created_at: str | None = None


class CurriculumResponse(BaseModel):
    id: str
    version: int
    curriculum_json: dict
    created_at: str


class TopicResponse(BaseModel):
    id: str
    title: str
    week_number: int
    day_number: int
    duration_minutes: int
    status: str
    scheduled_date: str | None = None


class ModuleResponse(BaseModel):
    id: str
    topic_id: str
    title: str
    content_markdown: str
    sources: list | None = None
    word_count: int | None = None
    estimated_read_minutes: int | None = None
    created_at: str | None = None


@router.post("/start", response_model=LearningSessionResponse)
async def start_learning_session(
    request: LearningStartRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start a new learning session.
    Creates a LearningSession in DB and triggers the PLA pipeline
    (Planner → Researcher → Composer) as a Celery background task.
    """
    service = LearningService(db)

    session = await service.create_session(
        user_id=current_user.id,
        topic=request.topic,
        level=request.level,
        duration_weeks=request.duration_weeks,
        hours_per_day=request.hours_per_day,
        language=request.language,
    )

    # Trigger background pipeline via Celery
    run_learning_pipeline.delay(
        session_id=str(session.id),
        user_id=str(current_user.id),
        config={
            "topic": request.topic,
            "duration_weeks": request.duration_weeks,
            "level": request.level,
            "hours_per_day": request.hours_per_day,
            "language": request.language,
        },
    )

    return LearningSessionResponse(
        id=str(session.id),
        topic=session.topic,
        level=session.level,
        duration_weeks=session.duration_weeks,
        hours_per_day=session.hours_per_day,
        language=session.language,
        status=session.status,
        created_at=session.created_at.isoformat() if session.created_at else None,
    )


@router.get("/sessions", response_model=list[LearningSessionResponse])
async def list_learning_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all learning sessions for the current user."""
    service = LearningService(db)
    sessions = await service.get_sessions(current_user.id)
    return [
        LearningSessionResponse(
            id=str(s.id),
            topic=s.topic,
            level=s.level,
            duration_weeks=s.duration_weeks,
            hours_per_day=s.hours_per_day,
            language=s.language,
            status=s.status,
            created_at=s.created_at.isoformat() if s.created_at else None,
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=LearningSessionResponse)
async def get_learning_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific learning session."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")
    return LearningSessionResponse(
        id=str(session.id),
        topic=session.topic,
        level=session.level,
        duration_weeks=session.duration_weeks,
        hours_per_day=session.hours_per_day,
        language=session.language,
        status=session.status,
        created_at=session.created_at.isoformat() if session.created_at else None,
    )


@router.get("/{session_id}/curriculum", response_model=CurriculumResponse)
async def get_session_curriculum(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the curriculum for a learning session."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")
    curriculum = await service.get_curriculum(session_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found yet (still processing)")
    return CurriculumResponse(
        id=str(curriculum.id),
        version=curriculum.version,
        curriculum_json=curriculum.curriculum_json,
        created_at=curriculum.created_at.isoformat() if curriculum.created_at else None,
    )


@router.get("/{session_id}/topics", response_model=list[TopicResponse])
async def get_session_topics(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all topics for a learning session."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")
    topics = await service.get_topics(session_id)
    return [
        TopicResponse(
            id=t.id,
            title=t.title,
            week_number=t.week_number,
            day_number=t.day_number,
            duration_minutes=t.duration_minutes,
            status=t.status,
            scheduled_date=t.scheduled_date.isoformat() if t.scheduled_date else None,
        )
        for t in topics
    ]


@router.get("/{session_id}/modules/{topic_id}", response_model=ModuleResponse)
async def get_topic_module(
    session_id: UUID,
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the learning module for a specific topic."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")
    module = await service.get_module(topic_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return ModuleResponse(
        id=str(module.id),
        topic_id=module.topic_id,
        title=module.title,
        content_markdown=module.content_markdown,
        sources=module.sources,
        word_count=module.word_count,
        estimated_read_minutes=module.estimated_read_minutes,
        created_at=module.created_at.isoformat() if module.created_at else None,
    )


@router.patch("/{session_id}/topics/{topic_id}/complete")
async def complete_topic(
    session_id: UUID,
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"[COMPLETE] Marking topic {topic_id} as completed for session {session_id}")
    
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")
    
    topic = await service.update_topic_status(topic_id, "completed")
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    logger.info(f"[COMPLETE] Topic {topic_id} status updated to: {topic.status}")
    
    next_topic = await service.activate_next_topic(session_id, topic_id)
    
    if next_topic:
        logger.info(f"[COMPLETE] Next topic found: {next_topic.id}, status: {next_topic.status}")
        generate_module_for_topic.delay(
            session_id=str(session_id),
            user_id=str(current_user.id),
            topic_id=next_topic.id,
        )
    else:
        logger.warning(f"[COMPLETE] No next topic found after {topic_id}")
    
    return {
        "status": "ok",
        "topic_id": topic_id,
        "new_status": topic.status,
        "next_topic_id": next_topic.id if next_topic else None,
        "next_topic_status": next_topic.status if next_topic else None,
        "module_generating": next_topic.id if next_topic else None,
    }