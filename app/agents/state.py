from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

class LearningConfig(BaseModel):
    topic: str
    duration_weeks: int
    level: str
    hours_per_day: float
    language: str = "id"

class DaySchedule(BaseModel):
    day: int
    topic_id: str
    title: str
    duration_minutes: int
    status: str = "pending"
    search_queries: List[str]

class WeekSchedule(BaseModel):
    week: int
    title: str
    days: List[DaySchedule]

class Curriculum(BaseModel):
    curriculum_id: str
    topic: str
    total_weeks: int
    weeks: List[WeekSchedule]

class CourseLink(BaseModel):
    title: str
    platform: str           # coursera | udemy | edx | fastai | freecodecamp
    url: str
    instructor: Optional[str] = None
    rating: Optional[float] = None
    duration: Optional[str] = None
    price_type: str = "free"  # free | paid | audit
    description: str = ""
    relevant_section: str = ""  # bagian/week spesifik yang relevan dengan topik ini

class RawContent(BaseModel):
    source_type: str        # web | youtube | arxiv | semantic_scholar | wikipedia | pdf | course
    source_url: str
    source_title: str
    raw_text: str           # kosong jika embed_mode=False (sumber tipe course)
    topic_id: str
    relevance_score: float
    fetched_at: datetime
    # field untuk resource link
    embed_mode: bool = True             # True = masuk RAG Qdrant, False = hanya ditampilkan sebagai link
    display_url: Optional[str] = None   # URL yang ditampilkan ke user di modul
    course_metadata: Optional[CourseLink] = None  # hanya diisi jika source_type = course

class LearningModule(BaseModel):
    topic_id: str
    title: str
    content_markdown: str
    sources: List[Dict[str, str]]

class Message(BaseModel):
    role: str
    content: str

class QuizResult(BaseModel):
    topic_id: str
    score: float

class ProgressSignals(BaseModel):
    quiz_score: Optional[float] = None
    reading_time_ratio: Optional[float] = None
    question_frequency: Optional[float] = None
    self_assessment: Optional[float] = None
    material_rating: Optional[float] = None

class FeedbackAction(BaseModel):
    action: str
    topic_id: str

class AgentLog(BaseModel):
    timestamp: datetime
    agent: str
    level: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

class PLAState(TypedDict):
    user_id: str
    session_id: str
    learning_config: LearningConfig
    curriculum: Optional[Curriculum]
    research_results: List[RawContent]
    modules: List[LearningModule]
    chat_history: List[Message]
    quiz_results: List[QuizResult]
    mastery_scores: Dict[str, float]
    progress_signals: Optional[ProgressSignals]
    feedback_actions: List[FeedbackAction]
    agent_logs: List[AgentLog]
    concept_graph: Optional[Dict[str, Any]]
