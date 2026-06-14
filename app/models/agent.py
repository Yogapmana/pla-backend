import uuid
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.db.database import Base

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False)
    topic_id = Column(String(100), ForeignKey("topics.id"), nullable=True)
    role = Column(String(20), nullable=False)
    content = Column(String, nullable=False)
    sources = Column(JSONB, nullable=True)
    rag_faithfulness = Column(Float, nullable=True)
    rag_answer_relevancy = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class QuizResult(Base):
    __tablename__ = "quiz_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False)
    topic_id = Column(String(100), ForeignKey("topics.id"), nullable=True)
    attempt_number = Column(Integer, default=1)
    score = Column(Float, nullable=False)
    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    answers_detail = Column(JSONB, nullable=True)
    time_spent_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ProgressSignal(Base):
    __tablename__ = "progress_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False)
    topic_id = Column(String(100), ForeignKey("topics.id"), nullable=True)
    signal_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    metadata_json = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AgentLog(Base):
    """
    Real-time activity log for the LangGraph pipeline.
    One row per (agent, level, message) emitted during a pipeline run.

    Used by:
      - Backend: write_log() in run_orchestrator.py persists each
        log emitted by the planner/researcher/composer/replan nodes
      - Frontend: AgentLog page reads these rows to show live activity
        via the WebSocket
    """
    __tablename__ = "agent_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False)
    agent = Column(String(50), nullable=False)  # planner | researcher | composer | feedback | tutor
    level = Column(String(20), nullable=False, default="info")  # info | warn | error | success
    message = Column(String, nullable=False)
    metadata_json = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

