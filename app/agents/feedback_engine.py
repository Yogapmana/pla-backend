from datetime import datetime
from app.agents.state import ProgressSignals, FeedbackAction
import logging

logger = logging.getLogger(__name__)

def evaluate_mastery(score: float, topic_id: str) -> FeedbackAction:
    """
    Menentukan aksi yang harus dilakukan Planner berdasarkan mastery score.
    """
    if score < 0.60:
        action = "repeat"
    elif score < 0.75:
        action = "review"
    elif score <= 0.90:
        action = "continue"
    else:
        action = "accelerate"
        
    logger.info(f"[FEEDBACK] Topic: {topic_id} | Score: {score:.2f} -> Action: {action}")
    return FeedbackAction(action=action, topic_id=topic_id)

def calculate_mastery(signals: ProgressSignals) -> float:
    """
    Menghitung mastery score (0.0 - 1.0) dari 5 sinyal kemajuan.
    Semua sinyal diasumsikan sudah ternormalisasi (0.0 - 1.0) di level API.
    """
    # Mengambil nilai default jika None
    quiz_score = signals.quiz_score if signals.quiz_score is not None else 0.5
    reading_time_ratio = signals.reading_time_ratio if signals.reading_time_ratio is not None else 0.5
    question_freq = signals.question_frequency if signals.question_frequency is not None else 0.5
    self_assess = signals.self_assessment if signals.self_assessment is not None else 0.5
    material_rating = signals.material_rating if signals.material_rating is not None else 0.5

    # Formula sesuai PRD
    score = (
        quiz_score * 0.40 +
        reading_time_ratio * 0.20 +
        question_freq * 0.20 +
        self_assess * 0.15 +
        material_rating * 0.05
    )
    
    # Batasi agar score tidak lebih dari 1.0 dan tidak kurang dari 0.0
    return max(0.0, min(1.0, score))

def run_feedback_loop(signals: ProgressSignals, topic_id: str) -> tuple[float, FeedbackAction]:
    """
    Fungsi utama yang dipanggil oleh API untuk menjalankan feedback loop.
    Mengembalikan mastery score dan aksi yang harus diambil.
    """
    score = calculate_mastery(signals)
    action = evaluate_mastery(score, topic_id)
    return score, action
