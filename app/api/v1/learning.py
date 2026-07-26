from uuid import UUID
import uuid
import logging
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form, File, UploadFile, status
from typing import List
from fastapi_cache.decorator import cache
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.db.database import get_db, async_sessionmaker
from app.dependencies import get_current_user
from app.models.user import User
from app.models.learning import LearningSession, Topic, LearningModule

logger = logging.getLogger(__name__)

import fitz  # PyMuPDF


from app.services.learning_service import LearningService
from app.tasks.run_orchestrator import run_learning_pipeline, resume_learning_pipeline
from app.tasks.generate_module import generate_module_for_topic
from app.tasks.email_tasks import send_progress_email_task

router = APIRouter()


class LearningStartRequest(BaseModel):
    topic: str
    duration_weeks: int
    level: str
    hours_per_day: float
    language: str = "id"


class LanguageUpdateRequest(BaseModel):
    language: str


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
    completed_at: str | None = None
    feedback_action: str | None = None
    mastery_score: float | None = None
    quiz_score: float | None = None
    has_remedial: bool = False
    has_deep_dive: bool = False


class ModuleResponse(BaseModel):
    id: str
    topic_id: str
    title: str
    content_markdown: str
    remedial_markdown: str | None = None
    deep_dive_markdown: str | None = None
    content_version: int = 1
    sources: list | None = None
    word_count: int | None = None
    estimated_read_minutes: int | None = None
    created_at: str | None = None


class AgentLogResponse(BaseModel):
    id: str
    agent: str
    level: str
    message: str
    metadata: dict | None = None
    created_at: str | None = None


@router.post("/start", response_model=LearningSessionResponse)
async def start_learning_session(
    topic: str = Form(...),
    duration_weeks: int = Form(...),
    level: str = Form(...),
    hours_per_day: float = Form(...),
    language: str = Form("id"),
    files: List[UploadFile] = File(default=[]),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start a new learning session.
    Creates a LearningSession in DB and triggers the Synapsa pipeline
    (Planner → Researcher → Composer) as a Celery background task.
    """
    service = LearningService(db)

    # Check session limit
    existing_sessions = await service.get_sessions(current_user.id)
    if len(existing_sessions) >= 3:
        raise HTTPException(
            status_code=400, 
            detail="Batas maksimal sesi belajar tercapai (maksimal 3). Selesaikan atau hapus sesi yang ada terlebih dahulu."
        )

    import io
    import docx

    # Process uploaded files (validate and extract text)
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maksimal 3 file yang diizinkan.")

    context_text = ""
    MAX_SIZE = 10 * 1024 * 1024

    for file in files:
        if not file.filename:
            continue
            
        try:
            content = await file.read()
            if len(content) > MAX_SIZE:
                raise HTTPException(status_code=400, detail=f"File {file.filename} melebihi batas 10MB.")

            filename_lower = file.filename.lower()
            if filename_lower.endswith(".pdf"):
                doc = fitz.open(stream=content, filetype="pdf")
                for page in doc:
                    context_text += page.get_text() + "\n\n"
                doc.close()
            elif filename_lower.endswith(".docx"):
                doc = docx.Document(io.BytesIO(content))
                for para in doc.paragraphs:
                    context_text += para.text + "\n"
            elif filename_lower.endswith(".txt"):
                context_text += content.decode("utf-8") + "\n\n"
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error parsing file {file.filename}: {e}")

    # Limit context_text to avoid hitting payload too large errors
    if len(context_text) > 15000:
        context_text = context_text[:15000] + "\n...[TRUNCATED]"

    session = await service.create_session(
        user_id=current_user.id,
        topic=topic,
        level=level,
        duration_weeks=duration_weeks,
        hours_per_day=hours_per_day,
        language=language,
    )

    # Trigger background pipeline via Celery
    run_learning_pipeline.delay(
        session_id=str(session.id),
        user_id=str(current_user.id),
        config={
            "topic": topic,
            "level": level,
            "duration_weeks": duration_weeks,
            "hours_per_day": hours_per_day,
            "language": language,
            "context_text": context_text if context_text else None
        }
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

@router.post("/{session_id}/resume")
async def resume_learning_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Resume a learning session that is waiting for approval."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")
        
    if session.status != "waiting_approval":
        raise HTTPException(status_code=400, detail=f"Session is not waiting for approval, current status: {session.status}")
        
    # Update status
    session.status = "processing"
    await db.commit()
    
    # Trigger resume task
    resume_learning_pipeline.delay(session_id=str(session_id))
    
    return {"status": "resumed", "session_id": str(session_id)}


@router.patch("/{session_id}/language")
async def update_session_language(
    session_id: UUID,
    request: LanguageUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the language preference for an ongoing learning session."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")
        
    session.language = request.language
    await db.commit()
    
    return {"status": "ok", "language": session.language}


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


@router.get("/{session_id}/agent-logs", response_model=list[AgentLogResponse])
async def get_session_agent_logs(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Persisted agent pipeline logs for a session.

    Used by the agent stream UI to restore progress after tab switch / refresh
    without waiting for the next live WebSocket event.
    """
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    logs = await service.get_agent_logs(session_id)
    return [
        AgentLogResponse(
            id=str(log.id),
            agent=log.agent,
            level=log.level,
            message=log.message,
            metadata=log.metadata_json,
            created_at=log.created_at.isoformat() if log.created_at else None,
        )
        for log in logs
    ]


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_learning_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a learning session owned by the current user."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")
    await service.delete_session(session_id)
    return None


@router.get("/{session_id}/curriculum", response_model=CurriculumResponse)
@cache(expire=300)
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
    
    from app.models.learning import LearningModule
    from sqlalchemy import select
    modules_res = await db.execute(
        select(LearningModule.topic_id, LearningModule.remedial_markdown, LearningModule.deep_dive_markdown)
        .where(LearningModule.session_id == session_id)
    )
    module_info = {row[0]: (row[1] is not None, row[2] is not None) for row in modules_res.all()}

    return [
        TopicResponse(
            id=t.id,
            title=t.title,
            week_number=t.week_number,
            day_number=t.day_number,
            duration_minutes=t.duration_minutes,
            status=t.status,
            scheduled_date=t.scheduled_date.isoformat() if t.scheduled_date else None,
            completed_at=t.completed_at.isoformat() if t.completed_at else None,
            feedback_action=t.feedback_action,
            mastery_score=t.mastery_score,
            quiz_score=t.quiz_score,
            has_remedial=module_info.get(t.id, (False, False))[0],
            has_deep_dive=module_info.get(t.id, (False, False))[1],
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
        remedial_markdown=module.remedial_markdown,
        deep_dive_markdown=module.deep_dive_markdown,
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
    
    topic_to_check = await service.get_topic(topic_id)
    if not topic_to_check:
        raise HTTPException(status_code=404, detail="Topic not found")
        
    # Validasi skor kuis minimal 80%
    if topic_to_check.quiz_score is None or topic_to_check.quiz_score < 0.80:
        raise HTTPException(status_code=403, detail="Anda harus lulus kuis (minimal 80%) untuk lanjut ke materi berikutnya.")
        
    topic = await service.update_topic_status(topic_id, "completed")
    
    logger.info(f"[COMPLETE] Topic {topic_id} status updated to: {topic.status}")

    # Send progress email asynchronously
    send_progress_email_task.delay(current_user.email, current_user.username, topic.title)

    next_topic = await service.activate_next_topic(session_id, topic_id)

    if next_topic:
        logger.info(f"[COMPLETE] Next topic found: {next_topic.id}, status: {next_topic.status}")
        # NOTE: Module generation for next topic is now triggered ONLY after quiz score >= 80%
        # See POST /quiz/submit endpoint - it calls trigger_next_module_after_quiz() when score >= 80%
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