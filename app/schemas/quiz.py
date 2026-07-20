from uuid import UUID
from typing import Literal
from pydantic import BaseModel

class QuizQuestion(BaseModel):
    question: str
    question_type: Literal["mcq", "true_false", "essay"] = "mcq"
    options: list[str] | None = None
    correct_answer: str
    explanation: str

class QuizResponse(BaseModel):
    quiz_id: str
    topic_id: str
    questions: list[QuizQuestion]
    total_questions: int
    time_limit_seconds: int | None = None

class QuizAnswer(BaseModel):
    question_index: int
    selected_answer: str

class QuizSubmission(BaseModel):
    session_id: UUID
    topic_id: str
    quiz_id: str | None = None  # Required for cache-based grading (preferred)
    answers: list[QuizAnswer]
    time_spent_seconds: int | None = None
    questions_data: list[dict] | None = None  # Legacy fallback only

class QuizResultResponse(BaseModel):
    score: float
    total_questions: int
    correct_answers: int
    percentage: float
    feedback: str
    cooldown_remaining_seconds: int = 0
