from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from redis import asyncio as aioredis

from app.db.database import get_db
from app.config import settings
from app.dependencies import get_current_user
from app.models.user import User
from app.services.learning_service import LearningService
from app.tasks.generate_module import generate_module_for_topic

router = APIRouter(prefix="/modules", tags=["modules"])


class ModuleResponse(BaseModel):
    id: str
    topic_id: str
    session_id: str
    title: str
    content_markdown: str
    sources: list | None = None
    word_count: int | None = None
    estimated_read_minutes: int | None = None
    created_at: str | None = None


class ModuleStatusResponse(BaseModel):
    topic_id: str
    status: str
    module_exists: bool


@router.get("/{topic_id}", response_model=ModuleResponse)
async def get_module(
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the learning module for a specific topic.
    PRD: GET /api/v1/modules/{topic_id}
    """
    service = LearningService(db)
    module = await service.get_module(topic_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    return ModuleResponse(
        id=str(module.id),
        topic_id=module.topic_id,
        session_id=str(module.session_id),
        title=module.title,
        content_markdown=module.content_markdown,
        remedial_markdown=module.remedial_markdown,
        deep_dive_markdown=module.deep_dive_markdown,
        sources=module.sources,
        word_count=module.word_count,
        estimated_read_minutes=module.estimated_read_minutes,
        created_at=module.created_at.isoformat() if module.created_at else None,
    )


@router.get("/{topic_id}/status", response_model=ModuleStatusResponse)
async def get_module_status(
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get status of a topic and whether module exists.
    PRD: GET /api/v1/modules/{topic_id}/status
    """
    service = LearningService(db)
    topic = await service.get_topic(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    module = await service.get_module(topic_id)

    return ModuleStatusResponse(
        topic_id=topic_id,
        status=topic.status,
        module_exists=module is not None,
    )


@router.patch("/{topic_id}/complete")
async def complete_module(
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark a topic as completed.
    PRD: PATCH /api/v1/modules/{topic_id}/complete
    """
    service = LearningService(db)
    topic = await service.update_topic_status(topic_id, "completed")
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    return {
        "status": "ok",
        "topic_id": topic_id,
        "new_status": topic.status,
    }


class GenerateModuleResponse(BaseModel):
    status: str
    task_id: str | None = None
    module_id: str | None = None


@router.post("/{topic_id}/generate", response_model=GenerateModuleResponse)
async def trigger_generate_module(
    topic_id: str,
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = LearningService(db)
    
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    
    topic = await service.get_topic(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    module = await service.get_module(topic_id)
    if module:
        return GenerateModuleResponse(
            status="already_exists",
            module_id=str(module.id),
        )

    # Prevent React Strict Mode or frontend double-clicks from spawning duplicate Celery tasks
    redis = aioredis.from_url(settings.REDIS_URL)
    lock_key = f"lock:generate_module:{topic_id}"
    
    # Try to set a lock that expires in 600 seconds (10 minutes)
    # nx=True means it will only be set if it does not already exist
    is_acquired = await redis.set(lock_key, "locked", ex=600, nx=True)
    await redis.close()

    if not is_acquired:
        # A task is already generating this module
        return GenerateModuleResponse(
            status="generating",
            task_id="duplicate-prevented",
        )
    
    task = generate_module_for_topic.delay(
        session_id=str(session_id),
        user_id=str(current_user.id),
        topic_id=topic_id,
    )
    
    return GenerateModuleResponse(
        status="generating",
        task_id=task.id,
    )