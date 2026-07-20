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
    mindmap_json = Column(JSONB, nullable=True)  # Cached AI-generated Mermaid mind map
    concept_graph_json = Column(JSONB, nullable=True)  # Cached concept graph (root → clusters → concepts → topics → resources)
    enhanced_mindmap_json = Column(JSONB, nullable=True) # NotebookLM-style enhanced mindmap
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
    mastery_score = Column(Float, nullable=True)
    quiz_score = Column(Float, nullable=True)
    reading_time_ratio = Column(Float, nullable=True)
    question_frequency_score = Column(Float, nullable=True)
    self_assessment_score = Column(Float, nullable=True)
    material_rating_score = Column(Float, nullable=True)
    feedback_action = Column(String(100), nullable=True)

class LearningModule(Base):
    __tablename__ = "learning_modules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id = Column(String(100), ForeignKey("topics.id"), nullable=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id"), nullable=True)
    title = Column(String(500), nullable=False)
    content_markdown = Column(String, nullable=False)
    remedial_markdown = Column(String, nullable=True)
    deep_dive_markdown = Column(String, nullable=True)
    content_version = Column(Integer, default=1)
    sources = Column(JSONB, nullable=True)
    word_count = Column(Integer, nullable=True)
    estimated_read_minutes = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ResourceLink(Base):
    __tablename__ = "resource_links"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    module_id = Column(UUID(as_uuid=True), ForeignKey("learning_modules.id", ondelete="CASCADE"), nullable=True)
    topic_id = Column(String(100), ForeignKey("topics.id"), nullable=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("learning_sessions.id"), nullable=True)
    link_type = Column(String(20), nullable=False)          # source | course | video | paper
    title = Column(String(500), nullable=False)
    url = Column(String, nullable=False)
    platform = Column(String(100), nullable=True)           # coursera | udemy | edx | youtube | arxiv | fastai | freecodecamp
    price_type = Column(String(20), nullable=True)          # free | paid | audit | open_access
    rating = Column(Float, nullable=True)
    duration = Column(String(100), nullable=True)
    instructor = Column(String(200), nullable=True)
    description = Column(String, nullable=True)
    relevant_section = Column(String, nullable=True)        # section/week spesifik yang relevan
    embed_mode = Column(String(10), default="true")         # TRUE = embed ke RAG, FALSE = link saja
    click_count = Column(Integer, default=0)                # tracking engagement
    created_at = Column(DateTime(timezone=True), server_default=func.now())

