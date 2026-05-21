from pydantic import BaseModel


class UserMetricsResponse(BaseModel):
    session_id: str
    topic_progress: dict
    quiz_metrics: dict
    streak_days: int
    estimated_study_hours: float
    weekly_progress: list
