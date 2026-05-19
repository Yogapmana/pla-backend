import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.database import get_db
from app.models.agent import ProgressSignal as DBProgressSignal, MasteryScore as DBMasteryScore
from app.models.learning import LearningSession as DBLearningSession
from app.agents.state import ProgressSignals, PLAState
from app.agents.feedback_engine import run_feedback_loop
from app.agents.planner import replan_node
from pydantic import BaseModel
from typing import Optional, List

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
