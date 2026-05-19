from uuid import UUID
from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: UUID
    topic_id: str
    message: str
    include_sources: bool = True

class ChatSource(BaseModel):
    title: str
    type: str
    relevance: float = 0.0

class RAGMetrics(BaseModel):
    latency_ms: int
    chunks_used: int

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

    class Config:
        from_attributes = True
