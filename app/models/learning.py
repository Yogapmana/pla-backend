import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Date
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.db.database import Base

class LearningSession(Base):
    __tablename__ = "learning_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    topic = Column(String(500), nullable=False)
    level = Column(String(50), nullable=False)
    duration_weeks = Column(Integer, nullable=False)
    hours_per_day = Column(Float, nullable=False)
    language = Column(String(10), default="id")
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

class Curriculum(Base):
    __tablename__ = "curricula"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, default=1)
    curriculum_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Topic(Base):
    __tablename__ = "topics"

    id = Column(String(100), primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False)
    curriculum_id = Column(UUID(as_uuid=True), ForeignKey("curricula.id"), nullable=True)
    title = Column(String(500), nullable=False)
    week_number = Column(Integer, nullable=False)
    day_number = Column(Integer, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    status = Column(String(50), default="locked")
    scheduled_date = Column(Date, nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    search_queries = Column(JSONB, nullable=True)

class LearningModule(Base):
    __tablename__ = "learning_modules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id = Column(String(100), ForeignKey("topics.id"), nullable=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id"), nullable=True)
    title = Column(String(500), nullable=False)
    content_markdown = Column(String, nullable=False)
    content_version = Column(Integer, default=1)
    sources = Column(JSONB, nullable=True)
    word_count = Column(Integer, nullable=True)
    estimated_read_minutes = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
