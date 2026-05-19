from uuid import UUID
from pydantic import BaseModel

class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str

class QuizResponse(BaseModel):
    topic_id: str
    questions: list[QuizQuestion]
    total_questions: int

class QuizAnswer(BaseModel):
    question_index: int
    selected_answer: str

class QuizSubmission(BaseModel):
    session_id: UUID
    topic_id: str
    answers: list[QuizAnswer]
    time_spent_seconds: int | None = None

class QuizResultResponse(BaseModel):
    score: float
    total_questions: int
    correct_answers: int
    percentage: float
    feedback: str
