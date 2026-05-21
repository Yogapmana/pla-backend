from uuid import UUID
from pydantic import BaseModel
from datetime import datetime


class RAGMetricCreate(BaseModel):
    message_id: UUID | None = None
    session_id: UUID
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    answer_correctness: float | None = None
    latency_ms: int | None = None
    chunks_retrieved: int | None = None
    chunks_after_rerank: int | None = None


RAGMetricsCreate = RAGMetricCreate  # Alias for backwards compat


class RAGMetricResponse(BaseModel):
    id: str
    message_id: str | None
    session_id: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_recall: float | None
    context_precision: float | None
    answer_correctness: float | None
    latency_ms: int | None
    chunks_retrieved: int | None
    chunks_after_rerank: int | None
    created_at: str


class UXSurveyCreate(BaseModel):
    session_id: UUID
    ease_of_use: float | None = None
    material_relevance: float | None = None
    quiz_quality: float | None = None
    adaptivity_satisfaction: float | None = None
    overall_satisfaction: float | None = None
    open_feedback: str | None = None


class UXSurveyResponse(BaseModel):
    id: str
    session_id: str
    ease_of_use: float | None
    material_relevance: float | None
    quiz_quality: float | None
    adaptivity_satisfaction: float | None
    overall_satisfaction: float | None
    open_feedback: str | None
    submitted_at: str