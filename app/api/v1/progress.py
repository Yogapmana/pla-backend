import uuid
from collections import Counter, defaultdict
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.database import get_db
from app.models.agent import ProgressSignal as DBProgressSignal, MasteryScore as DBMasteryScore, QuizResult as DBQuizResult
from app.models.learning import LearningSession as DBLearningSession, Topic as DBTopic
from app.agents.state import ProgressSignals, PLAState
from app.agents.feedback_engine import run_feedback_loop
from app.agents.planner import replan_node
from app.schemas.progress import UserMetricsResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class SignalSubmit(BaseModel):
    session_id: str
    topic_id: str
    quiz_score: Optional[float] = None
    reading_time_ratio: Optional[float] = None
    question_frequency: Optional[float] = None
    self_assessment: Optional[float] = None
    material_rating: Optional[float] = None

class EvaluateResponse(BaseModel):
    mastery_score: float
    feedback_action: str
    message: str

class TopicUnlockStatusResponse(BaseModel):
    topic_id: str
    unlocked: bool
    latest_quiz_score: float | None = None


@router.get("/user-metrics/{session_id}", response_model=UserMetricsResponse)
async def get_user_metrics(session_id: str, db: AsyncSession = Depends(get_db)):
    session_uuid = uuid.UUID(session_id)

    session_result = await db.execute(
        select(DBLearningSession).where(DBLearningSession.id == session_uuid)
    )
    session = session_result.scalars().first()

    topic_rows = []
    quiz_rows = []
    learning_rows = []

    if session:
        topic_result = await db.execute(
            select(DBTopic).where(DBTopic.session_id == session_uuid)
        )
        topic_rows = topic_result.scalars().all()

        quiz_result = await db.execute(
            select(DBQuizResult).where(DBQuizResult.session_id == session_uuid)
        )
        quiz_rows = quiz_result.scalars().all()

        learning_result = await db.execute(
            select(DBLearningSession).where(DBLearningSession.user_id == session.user_id)
        )
        learning_rows = learning_result.scalars().all()

    total_topics = len(topic_rows)
    completed_topics = sum(1 for topic in topic_rows if topic.status == "completed")
    status_counts = Counter(topic.status for topic in topic_rows)
    completion_percentage = round((completed_topics / total_topics) * 100, 2) if total_topics else 0.0

    quiz_scores = [quiz.score for quiz in quiz_rows if quiz.score is not None]
    average_score = round(sum(quiz_scores) / len(quiz_scores), 2) if quiz_scores else 0.0

    estimated_study_minutes = sum((topic.duration_minutes or 0) for topic in topic_rows)
    estimated_study_hours = round(estimated_study_minutes / 60.0, 2)

    activity_dates = set()
    for learning_session in learning_rows:
        active_at = learning_session.completed_at or learning_session.created_at
        if active_at:
            activity_dates.add(active_at.date())

    streak_days = 0
    if activity_dates:
        cursor = max(activity_dates)
        while cursor in activity_dates:
            streak_days += 1
            cursor = cursor - timedelta(days=1)

    today = date.today()
    weekly_counts = defaultdict(int)
    for topic in topic_rows:
        if topic.completed_at:
            completed_date = topic.completed_at.date()
            if today - timedelta(days=6) <= completed_date <= today:
                weekly_counts[completed_date.isoformat()] += 1

    weekly_progress = [
        {"date": (today - timedelta(days=offset)).isoformat(), "completed": weekly_counts.get((today - timedelta(days=offset)).isoformat(), 0)}
        for offset in range(6, -1, -1)
    ]

    return UserMetricsResponse(
        session_id=session_id,
        topic_progress={
            "completed": completed_topics,
            "total": total_topics,
            "percentage": completion_percentage,
            "by_status": dict(status_counts),
        },
        quiz_metrics={
            "average_score": average_score,
            "total_quizzes": len(quiz_rows),
        },
        streak_days=streak_days,
        estimated_study_hours=estimated_study_hours,
        weekly_progress=weekly_progress,
    )

@router.post("/signal")
async def submit_progress_signals(data: SignalSubmit, db: AsyncSession = Depends(get_db)):
    """
    Simpan satu atau lebih progress signals ke dalam database.
    """
    signals = []
    
    # helper for mapping
    signal_map = {
        "quiz_score": data.quiz_score,
        "reading_time_ratio": data.reading_time_ratio,
        "question_frequency": data.question_frequency,
        "self_assessment": data.self_assessment,
        "material_rating": data.material_rating
    }
    
    for key, value in signal_map.items():
        if value is not None:
            new_signal = DBProgressSignal(
                id=uuid.uuid4(),
                session_id=uuid.UUID(data.session_id),
                topic_id=data.topic_id,
                signal_type=key,
                value=value
            )
            db.add(new_signal)
            signals.append(key)
            
    if not signals:
        raise HTTPException(status_code=400, detail="No signals provided")
        
    await db.commit()
    return {"status": "success", "message": f"Saved signals: {', '.join(signals)}"}

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_feedback(session_id: str, topic_id: str, db: AsyncSession = Depends(get_db)):
    """
    Trigger Feedback Engine:
    1. Ambil semua progress signals untuk topic_id di session_id.
    2. Hitung mastery score.
    3. Tentukan FeedbackAction.
    4. Simpan MasteryScore ke database.
    5. Panggil replan_node untuk merevisi jadwal.
    """
    # Ambil sesi untuk mengecek state kurikulum
    result = await db.execute(select(DBLearningSession).where(DBLearningSession.id == uuid.UUID(session_id)))
    session = result.scalars().first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Ambil sinyal
    signal_result = await db.execute(
        select(DBProgressSignal)
        .where(DBProgressSignal.session_id == uuid.UUID(session_id))
        .where(DBProgressSignal.topic_id == topic_id)
    )
    db_signals = signal_result.scalars().all()
    
    if not db_signals:
        # Jika belum ada sinyal, gunakan default atau tolak
        logger = __import__('logging').getLogger(__name__)
        logger.warning(f"No signals found for session {session_id}, topic {topic_id}")
        
    # Agregasi sinyal (ambil nilai terakhir per tipe)
    agg_signals = {}
    for s in db_signals:
        agg_signals[s.signal_type] = s.value
        
    # Bentuk object state
    state_signals = ProgressSignals(
        quiz_score=agg_signals.get("quiz_score"),
        reading_time_ratio=agg_signals.get("reading_time_ratio"),
        question_frequency=agg_signals.get("question_frequency"),
        self_assessment=agg_signals.get("self_assessment"),
        material_rating=agg_signals.get("material_rating")
    )
    
    # Jalankan feedback loop murni (tanpa LLM)
    mastery_score, feedback_action = run_feedback_loop(state_signals, topic_id)
    
    # Simpan ke DB
    db_mastery = DBMasteryScore(
        id=uuid.uuid4(),
        session_id=uuid.UUID(session_id),
        topic_id=topic_id,
        mastery_score=mastery_score,
        quiz_score=state_signals.quiz_score,
        reading_time_ratio=state_signals.reading_time_ratio,
        question_frequency_score=state_signals.question_frequency,
        self_assessment_score=state_signals.self_assessment,
        material_rating_score=state_signals.material_rating,
        feedback_action=feedback_action.action
    )
    db.add(db_mastery)
    
    # Lakukan Replanning jika ada kurikulum
    message = "Mastery evaluated. No replanning performed (no existing curriculum)."
    
    from app.models.learning import Curriculum as DBCurriculum
    curr_result = await db.execute(
        select(DBCurriculum)
        .where(DBCurriculum.session_id == uuid.UUID(session_id))
        .order_by(DBCurriculum.version.desc())
    )
    db_curriculum = curr_result.scalars().first()
    
    if db_curriculum and db_curriculum.curriculum_json:
        try:
            from app.agents.state import Curriculum
            # Rekonstruksi state
            curriculum_obj = Curriculum.model_validate(db_curriculum.curriculum_json)
            
            # Buat dummy state
            state: PLAState = {
                "user_id": str(session.user_id),
                "session_id": str(session.id),
                "learning_config": None, # tidak terlalu butuh untuk replan jika curriculum ada
                "curriculum": curriculum_obj,
                "research_results": [],
                "modules": [],
                "chat_history": [],
                "quiz_results": [],
                "mastery_scores": {topic_id: mastery_score},
                "progress_signals": state_signals,
                "feedback_actions": [feedback_action],
                "agent_logs": []
            }
            
            # Panggil replan_node
            new_state = replan_node(state)
            
            # Simpan kurikulum yang baru sebagai versi baru
            if new_state["curriculum"]:
                new_db_curr = DBCurriculum(
                    id=uuid.uuid4(),
                    session_id=uuid.UUID(session_id),
                    version=db_curriculum.version + 1,
                    curriculum_json=new_state["curriculum"].model_dump()
                )
                db.add(new_db_curr)
                message = f"Mastery evaluated. Curriculum revised based on action: {feedback_action.action}"
                
        except Exception as e:
            message = f"Mastery evaluated but replanning failed: {str(e)}"
            
    await db.commit()
    
    return EvaluateResponse(
        mastery_score=mastery_score,
        feedback_action=feedback_action.action,
        message=message
    )

@router.get("/topic-unlock/{session_id}/{topic_id}", response_model=TopicUnlockStatusResponse)
async def get_topic_unlock_status(
    session_id: str,
    topic_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Cek apakah topik sudah unlocked berdasarkan nilai kuis minimum 80%.
    """
    session_uuid = uuid.UUID(session_id)

    result = await db.execute(
        select(DBQuizResult)
        .where(DBQuizResult.session_id == session_uuid)
        .where(DBQuizResult.topic_id == topic_id)
        .order_by(DBQuizResult.created_at.desc())
    )
    latest_quiz = result.scalars().first()

    latest_score = latest_quiz.score if latest_quiz else None
    unlocked = latest_score is not None and (latest_score * 100) >= 80.0

    return TopicUnlockStatusResponse(
        topic_id=topic_id,
        unlocked=unlocked,
        latest_quiz_score=round(latest_score * 100, 1) if latest_score is not None else None,
    )
