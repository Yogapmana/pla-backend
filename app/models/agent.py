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

class MasteryScore(Base):
    __tablename__ = "mastery_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False)
    topic_id = Column(String(100), ForeignKey("topics.id"), nullable=True)
    mastery_score = Column(Float, nullable=False)
    quiz_score = Column(Float, nullable=True)
    reading_time_ratio = Column(Float, nullable=True)
    question_frequency_score = Column(Float, nullable=True)
    self_assessment_score = Column(Float, nullable=True)
    material_rating_score = Column(Float, nullable=True)
    feedback_action = Column(String(100), nullable=True)
    calculated_at = Column(DateTime(timezone=True), server_default=func.now())

class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=True)
    agent = Column(String(50), nullable=False)
    level = Column(String(20), default="info")
    message = Column(String, nullable=False)
    metadata_json = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class RAGMetric(Base):
    __tablename__ = "rag_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id"), nullable=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id"), nullable=True)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    answer_correctness = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    chunks_retrieved = Column(Integer, nullable=True)
    chunks_after_rerank = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UXSurvey(Base):
    __tablename__ = "ux_surveys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    ease_of_use = Column(Float, nullable=True)
    material_relevance = Column(Float, nullable=True)
    quiz_quality = Column(Float, nullable=True)
    adaptivity_satisfaction = Column(Float, nullable=True)
    overall_satisfaction = Column(Float, nullable=True)
    open_feedback = Column(String, nullable=True)
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())
