from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.agent import RAGMetric, UXSurvey
from app.services.learning_service import LearningService
from app.schemas.metrics import (
    RAGMetricCreate, RAGMetricResponse,
    UXSurveyCreate, UXSurveyResponse,
)

router = APIRouter()


# ─── RAG Metrics ────────────────────────────────────────────

@router.post("/rag", response_model=RAGMetricResponse)
async def record_rag_metric(
    metric: RAGMetricCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Record a RAG evaluation metric after a chat interaction."""
    db_metric = RAGMetric(
        message_id=metric.message_id,
        session_id=metric.session_id,
        faithfulness=metric.faithfulness,
        answer_relevancy=metric.answer_relevancy,
        context_recall=metric.context_recall,
        context_precision=metric.context_precision,
        answer_correctness=metric.answer_correctness,
        latency_ms=metric.latency_ms,
        chunks_retrieved=metric.chunks_retrieved,
        chunks_after_rerank=metric.chunks_after_rerank,
    )
    db.add(db_metric)
    await db.commit()
    await db.refresh(db_metric)
    return RAGMetricResponse(
        id=str(db_metric.id),
        message_id=str(db_metric.message_id) if db_metric.message_id else None,
        session_id=str(db_metric.session_id),
        faithfulness=db_metric.faithfulness,
        answer_relevancy=db_metric.answer_relevancy,
        context_recall=db_metric.context_recall,
        context_precision=db_metric.context_precision,
        answer_correctness=db_metric.answer_correctness,
        latency_ms=db_metric.latency_ms,
        chunks_retrieved=db_metric.chunks_retrieved,
        chunks_after_rerank=db_metric.chunks_after_rerank,
        created_at=db_metric.created_at.isoformat() if db_metric.created_at else None,
    )


@router.get("/rag/{session_id}", response_model=list[RAGMetricResponse])
async def get_rag_metrics(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all RAG metrics for a session."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(RAGMetric)
        .where(RAGMetric.session_id == session_id)
        .order_by(RAGMetric.created_at.desc())
    )
    metrics = result.scalars().all()
    return [
        RAGMetricResponse(
            id=str(m.id),
            message_id=str(m.message_id) if m.message_id else None,
            session_id=str(m.session_id),
            faithfulness=m.faithfulness,
            answer_relevancy=m.answer_relevancy,
            context_recall=m.context_recall,
            context_precision=m.context_precision,
            answer_correctness=m.answer_correctness,
            latency_ms=m.latency_ms,
            chunks_retrieved=m.chunks_retrieved,
            chunks_after_rerank=m.chunks_after_rerank,
            created_at=m.created_at.isoformat() if m.created_at else None,
        )
        for m in metrics
    ]


# ─── UX Survey ───────────────────────────────────────────────

@router.post("/ux", response_model=UXSurveyResponse)
async def submit_ux_survey(
    survey: UXSurveyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit a UX survey response."""
    service = LearningService(db)
    session = await service.get_session(survey.session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    db_survey = UXSurvey(
        session_id=survey.session_id,
        user_id=current_user.id,
        ease_of_use=survey.ease_of_use,
        material_relevance=survey.material_relevance,
        quiz_quality=survey.quiz_quality,
        adaptivity_satisfaction=survey.adaptivity_satisfaction,
        overall_satisfaction=survey.overall_satisfaction,
        open_feedback=survey.open_feedback,
    )
    db.add(db_survey)
    await db.commit()
    await db.refresh(db_survey)
    return UXSurveyResponse(
        id=str(db_survey.id),
        session_id=str(db_survey.session_id),
        ease_of_use=db_survey.ease_of_use,
        material_relevance=db_survey.material_relevance,
        quiz_quality=db_survey.quiz_quality,
        adaptivity_satisfaction=db_survey.adaptivity_satisfaction,
        overall_satisfaction=db_survey.overall_satisfaction,
        open_feedback=db_survey.open_feedback,
        submitted_at=db_survey.submitted_at.isoformat() if db_survey.submitted_at else None,
    )


@router.get("/ux/{session_id}", response_model=list[UXSurveyResponse])
async def get_ux_surveys(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all UX surveys for a session."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(UXSurvey)
        .where(UXSurvey.session_id == session_id)
        .order_by(UXSurvey.submitted_at.desc())
    )
    surveys = result.scalars().all()
    return [
        UXSurveyResponse(
            id=str(s.id),
            session_id=str(s.session_id),
            ease_of_use=s.ease_of_use,
            material_relevance=s.material_relevance,
            quiz_quality=s.quiz_quality,
            adaptivity_satisfaction=s.adaptivity_satisfaction,
            overall_satisfaction=s.overall_satisfaction,
            open_feedback=s.open_feedback,
            submitted_at=s.submitted_at.isoformat() if s.submitted_at else None,
        )
        for s in surveys
    ]