from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.learning_service import LearningService

router = APIRouter(prefix="/curriculum", tags=["curriculum"])


class CurriculumResponse(BaseModel):
    id: str
    session_id: str
    version: int
    curriculum_json: dict
    created_at: str | None = None


class TopicResponse(BaseModel):
    id: str
    title: str
    week_number: int
    day_number: int
    duration_minutes: int
    status: str
    search_queries: list | None = None
    scheduled_date: str | None = None


class CurriculumDetailResponse(BaseModel):
    curriculum: CurriculumResponse | None = None
    topics: list[TopicResponse] = []


@router.get("/{session_id}", response_model=CurriculumResponse)
async def get_curriculum(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the curriculum for a learning session.
    PRD: GET /api/v1/curriculum/{session_id}
    """
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    curriculum = await service.get_curriculum(session_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found yet (still processing)")

    return CurriculumResponse(
        id=str(curriculum.id),
        session_id=str(curriculum.session_id),
        version=curriculum.version,
        curriculum_json=curriculum.curriculum_json,
        created_at=curriculum.created_at.isoformat() if curriculum.created_at else None,
    )


@router.get("/{session_id}/detail", response_model=CurriculumDetailResponse)
async def get_curriculum_detail(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get curriculum with all topics included."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    curriculum = await service.get_curriculum(session_id)
    topics = await service.get_topics(session_id)

    curriculum_resp = None
    if curriculum:
        curriculum_resp = CurriculumResponse(
            id=str(curriculum.id),
            session_id=str(curriculum.session_id),
            version=curriculum.version,
            curriculum_json=curriculum.curriculum_json,
            created_at=curriculum.created_at.isoformat() if curriculum.created_at else None,
        )

    topic_list = [
        TopicResponse(
            id=t.id,
            title=t.title,
            week_number=t.week_number,
            day_number=t.day_number,
            duration_minutes=t.duration_minutes,
            status=t.status,
            search_queries=t.search_queries,
            scheduled_date=t.scheduled_date.isoformat() if t.scheduled_date else None,
        )
        for t in topics
    ]

    return CurriculumDetailResponse(
        curriculum=curriculum_resp,
        topics=topic_list,
    )