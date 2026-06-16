from uuid import UUID
from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: UUID
    topic_id: str | None = None
    message: str
    include_sources: bool = True

class ChatSource(BaseModel):
    title: str
    type: str
    relevance: float = 0.0

class RAGMetrics(BaseModel):
    latency_ms: int
    chunks_used: int


class RAGASSummary(BaseModel):
    """Aggregated RAGAS quality metrics for a session — for the dashboard widget."""
    session_id: str
    total_messages: int
    scored_messages: int
    avg_faithfulness: float | None = None
    avg_answer_relevancy: float | None = None
    p95_faithfulness: float | None = None
    p95_answer_relevancy: float | None = None
    flagged_messages: int = 0  # messages with either score < 0.5

class ChatResponse(BaseModel):
    message_id: str | None = None
    response: str
    sources: list[ChatSource] = []
    rag_metrics: RAGMetrics

class ChatHistoryMessage(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    # RAGAS evaluation scores (Phase polish #1) — None until RAGAS has scored
    rag_faithfulness: float | None = None
    rag_answer_relevancy: float | None = None
    sources: list["ChatSource"] = []

    class Config:
        from_attributes = True

class ChatSessionResponse(BaseModel):
    id: str
    topic: str
    created_at: str

class ChatSessionCreate(BaseModel):
    title: str = "Percakapan Baru"
