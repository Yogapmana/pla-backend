import uuid
import logging
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.database import get_db
from app.dependencies import get_current_user, verify_session_owner, verify_topic_owner
from app.models.agent import ProgressSignal as DBProgressSignal, QuizResult as DBQuizResult
from app.models.learning import LearningSession as DBLearningSession, Topic as DBTopic
from app.models.user import User
from app.agents.state import ProgressSignals
from app.agents.feedback_engine import run_feedback_loop
from app.schemas.progress import UserMetricsResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class SignalSubmit(BaseModel):
    session_id: str
    topic_id: str
    quiz_score: Optional[float] = None
    reading_time_ratio: Optional[float] = None
    reading_time_seconds: Optional[float] = None
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
async def get_user_metrics(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Ownership check — raises 404 if missing or owned by another user,
    # 400 if the id is malformed.
    session = await verify_session_owner(session_id, current_user, db)
    session_uuid = session.id

    from app.services import metrics_cache
    cached = await metrics_cache.get_user_metrics(str(session_uuid))
    if cached is not None:
        return UserMetricsResponse(**cached)

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

    response = UserMetricsResponse(
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
    await metrics_cache.set_user_metrics(str(session_uuid), response.model_dump())
    return response

@router.post("/signal")
async def submit_progress_signals(
    data: SignalSubmit,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Simpan satu atau lebih progress signals ke dalam database.

    Ownership is enforced twice: the session must belong to the caller, and
    the topic must belong to that same session — so a user cannot write
    signals against another user's (session, topic) pair.
    """
    session = await verify_session_owner(data.session_id, current_user, db)
    await verify_topic_owner(
        data.topic_id, current_user, db, require_session_id=session.id
    )

    signals = []

    # helper for mapping
    signal_map = {
        "quiz_score": data.quiz_score,
        "reading_time_ratio": data.reading_time_ratio,
        "reading_time_seconds": data.reading_time_seconds,
        "question_frequency": data.question_frequency,
        "self_assessment": data.self_assessment,
        "material_rating": data.material_rating
    }

    for key, value in signal_map.items():
        if value is not None:
            new_signal = DBProgressSignal(
                id=uuid.uuid4(),
                session_id=session.id,
                topic_id=data.topic_id,
                signal_type=key,
                value=value
            )
            db.add(new_signal)
            signals.append(key)

    if not signals:
        raise HTTPException(status_code=400, detail="No signals provided")

    await db.commit()
    from app.services.metrics_cache import invalidate_session_metrics
    await invalidate_session_metrics(str(session.id))
    return {"status": "success", "message": f"Saved signals: {', '.join(signals)}"}

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_feedback(
    session_id: str,
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger Feedback Engine:
    1. Ambil semua progress signals untuk topic_id di session_id.
    2. Hitung mastery score.
    3. Tentukan FeedbackAction (remedial / continue / enrichment).
    4. Simpan mastery + signal breakdown ke topic row.

    Curriculum replan is intentionally NOT performed here. Adaptive
    content is delivered as supplementary modules (remedial / deep-dive)
    via the quiz post-eval path.
    """
    # Ownership: session + topic must both belong to the caller (and the
    # topic must be scoped to this session). Guards run before any reads,
    # replacing the previous soft "if not session" check that leaked the
    # existence of other users' sessions.
    session = await verify_session_owner(session_id, current_user, db)
    session_uuid = session.id
    await verify_topic_owner(topic_id, current_user, db, require_session_id=session_uuid)

    # Ambil sinyal
    signal_result = await db.execute(
        select(DBProgressSignal)
        .where(DBProgressSignal.session_id == session_uuid)
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

    # Simpan ke DB (Update Topic)
    topic_result = await db.execute(select(DBTopic).where(DBTopic.id == topic_id))
    db_topic = topic_result.scalars().first()

    if db_topic:
        db_topic.mastery_score = mastery_score
        db_topic.quiz_score = state_signals.quiz_score
        db_topic.reading_time_ratio = state_signals.reading_time_ratio
        db_topic.question_frequency_score = state_signals.question_frequency
        db_topic.self_assessment_score = state_signals.self_assessment
        db_topic.material_rating_score = state_signals.material_rating
        db_topic.feedback_action = feedback_action.action

    # Fetch context for remedial action
    if feedback_action.action == "remedial":
        from app.models.agent import QuizResult as DBQuizResult
        from app.models.learning import LearningModule as DBModule
        
        quiz_res = await db.execute(
            select(DBQuizResult)
            .where(DBQuizResult.session_id == session_uuid)
            .where(DBQuizResult.topic_id == topic_id)
            .order_by(DBQuizResult.created_at.desc())
        )
        latest_quiz = quiz_res.scalars().first()
        
        mod_res = await db.execute(
            select(DBModule)
            .where(DBModule.session_id == session_uuid)
            .where(DBModule.topic_id == topic_id)
        )
        db_mod = mod_res.scalars().first()
        
        wrong_context = []
        if latest_quiz and latest_quiz.answers_detail:
            for detail in latest_quiz.answers_detail:
                if not detail.get("is_correct"):
                    q_text = detail.get("question_text", f"Topik soal ke-{detail.get('question_index', 0)+1}")
                    wrong_context.append(f"Q: {q_text}")
        
        if wrong_context:
            feedback_action.context = "Topik remedial harus difokuskan pada perbaikan pemahaman pertanyaan berikut: " + "; ".join(wrong_context)

    # Curriculum replan intentionally disabled — adaptive path is
    # remedial / deep-dive supplementary modules only (see quiz post-eval).
    await db.commit()
    from app.services.metrics_cache import invalidate_session_metrics
    await invalidate_session_metrics(str(session_uuid))

    return EvaluateResponse(
        mastery_score=mastery_score,
        feedback_action=feedback_action.action,
        message="Mastery evaluated.",
    )

@router.get("/topic-unlock/{session_id}/{topic_id}", response_model=TopicUnlockStatusResponse)
async def get_topic_unlock_status(
    session_id: str,
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Cek apakah topik sudah unlocked berdasarkan nilai kuis minimum 80%.
    """
    # Ownership: both the session and the (session-scoped) topic must belong
    # to the caller before we expose quiz results for them.
    session = await verify_session_owner(session_id, current_user, db)
    await verify_topic_owner(topic_id, current_user, db, require_session_id=session.id)

    result = await db.execute(
        select(DBQuizResult)
        .where(DBQuizResult.session_id == session.id)
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


@router.get("/{session_id}/daily-study-time")
async def get_daily_study_time(
    session_id: str,
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return daily study time (in minutes) for the last N days.

    Aggregates two sources:
      1. ``reading_time_seconds`` signals from progress_signals table
         (recorded by the frontend ReadingTracker when the user reads a module)
      2. ``time_spent_seconds`` from quiz_results table
         (recorded when the user completes a quiz)

    Returns a list of {date, reading_minutes, quiz_minutes, total_minutes}
    sorted ascending by date.
    """
    logger = logging.getLogger(__name__)
    session = await verify_session_owner(session_id, current_user, db)
    session_uuid = session.id

    from app.services import metrics_cache
    cached = await metrics_cache.get_daily_study_time(str(session_uuid), days)
    if cached is not None:
        return cached

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # 1. Reading time from progress signals
    reading_result = await db.execute(
        select(DBProgressSignal)
        .where(DBProgressSignal.session_id == session_uuid)
        .where(DBProgressSignal.signal_type == "reading_time_seconds")
        .where(DBProgressSignal.created_at >= cutoff)
    )
    reading_signals = reading_result.scalars().all()

    daily_reading: dict[str, float] = defaultdict(float)
    for sig in reading_signals:
        day_str = sig.created_at.date().isoformat()
        daily_reading[day_str] += sig.value  # seconds

    # 2. Quiz time from quiz results
    quiz_result = await db.execute(
        select(DBQuizResult)
        .where(DBQuizResult.session_id == session_uuid)
        .where(DBQuizResult.created_at >= cutoff)
    )
    quiz_rows = quiz_result.scalars().all()

    daily_quiz: dict[str, float] = defaultdict(float)
    for q in quiz_rows:
        if q.time_spent_seconds and q.created_at:
            day_str = q.created_at.date().isoformat()
            daily_quiz[day_str] += q.time_spent_seconds  # seconds

    # 3. Merge into a unified list
    all_dates = sorted(set(daily_reading.keys()) | set(daily_quiz.keys()))

    result = []
    for d in all_dates:
        reading_sec = daily_reading.get(d, 0)
        quiz_sec = daily_quiz.get(d, 0)
        result.append({
            "date": d,
            "reading_minutes": round(reading_sec / 60, 1),
            "quiz_minutes": round(quiz_sec / 60, 1),
            "total_minutes": round((reading_sec + quiz_sec) / 60, 1),
        })

    await metrics_cache.set_daily_study_time(str(session_uuid), days, result)
    return result

@router.get("/signals/{session_id}/{topic_id}")
async def get_topic_signals(
    session_id: str,
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the latest material_rating and self_assessment signals for a topic.
    Useful so the UI doesn't reset when the user reopens the module.
    """
    session = await verify_session_owner(session_id, current_user, db)
    await verify_topic_owner(
        topic_id, current_user, db, require_session_id=session.id
    )

    result = await db.execute(
        select(DBProgressSignal)
        .where(DBProgressSignal.session_id == session.id)
        .where(DBProgressSignal.topic_id == topic_id)
        .where(DBProgressSignal.signal_type.in_(["material_rating", "self_assessment"]))
        .order_by(DBProgressSignal.created_at.asc())
    )
    signals = result.scalars().all()
    
    # Use dictionary to keep only the latest value for each type (since we order by asc, latest overwrites)
    latest_signals = {}
    for sig in signals:
        latest_signals[sig.signal_type] = sig.value
        
    return latest_signals


@router.get("/{session_id}/recent-activity")
async def get_recent_activity(
    session_id: str,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Unified recent activity feed for the dashboard.

    Merges (newest first, capped by ``limit``):
      - quiz attempts
      - topic completions
      - reading sessions (reading_time_seconds signals, min 30s)
      - self-assessment / material rating
      - user chat messages (tutor + general)
    """
    from app.models.agent import ChatMessage as DBChatMessage

    session = await verify_session_owner(session_id, current_user, db)
    session_uuid = session.id
    limit = max(1, min(limit, 50))
    # Fetch extra per source so merge still has enough after filtering
    per_source = max(limit, 15)

    # Topic titles for enrichment
    topics_result = await db.execute(
        select(DBTopic).where(DBTopic.session_id == session_uuid)
    )
    topics = {t.id: t for t in topics_result.scalars().all()}

    events: list[dict] = []

    # 1. Quiz attempts
    quiz_rows = (
        await db.execute(
            select(DBQuizResult)
            .where(DBQuizResult.session_id == session_uuid)
            .order_by(DBQuizResult.created_at.desc())
            .limit(per_source)
        )
    ).scalars().all()
    for q in quiz_rows:
        if not q.created_at:
            continue
        title = topics.get(q.topic_id).title if q.topic_id and q.topic_id in topics else (q.topic_id or "Topik")
        score_pct = round(float(q.score) * 100, 1) if q.score is not None and q.score <= 1 else (
            round(float(q.score), 1) if q.score is not None else None
        )
        mins = round((q.time_spent_seconds or 0) / 60)
        events.append({
            "id": f"quiz-{q.id}",
            "type": "quiz",
            "title": f"Kuis: {title}",
            "description": (
                f"Skor: {int(score_pct) if score_pct is not None else '—'} — {mins} menit"
                if score_pct is not None
                else f"{mins} menit"
            ),
            "created_at": q.created_at.isoformat(),
            "score": score_pct,
            "topic_id": q.topic_id,
            "href": f"/progress/topic/{q.topic_id}" if q.topic_id else None,
        })

    # 2. Topic completions
    for t in topics.values():
        if t.status != "completed" or not t.completed_at:
            continue
        mastery = (
            round(float(t.mastery_score) * 100)
            if t.mastery_score is not None
            else None
        )
        events.append({
            "id": f"topic-{t.id}",
            "type": "topic",
            "title": f"Selesai: {t.title}",
            "description": (
                f"Mastery {mastery}%" if mastery is not None else "Topik diselesaikan"
            ),
            "created_at": t.completed_at.isoformat(),
            "score": mastery,
            "topic_id": t.id,
            "href": f"/module/{t.id}",
        })

    # 3. Reading signals (≥ 30 detik, agar tidak noise)
    reading_rows = (
        await db.execute(
            select(DBProgressSignal)
            .where(DBProgressSignal.session_id == session_uuid)
            .where(DBProgressSignal.signal_type == "reading_time_seconds")
            .where(DBProgressSignal.value >= 30)
            .order_by(DBProgressSignal.created_at.desc())
            .limit(per_source)
        )
    ).scalars().all()
    for sig in reading_rows:
        if not sig.created_at:
            continue
        title = (
            topics.get(sig.topic_id).title
            if sig.topic_id and sig.topic_id in topics
            else "Modul"
        )
        mins = max(1, round(float(sig.value) / 60))
        events.append({
            "id": f"read-{sig.id}",
            "type": "reading",
            "title": f"Membaca: {title}",
            "description": f"{mins} menit membaca",
            "created_at": sig.created_at.isoformat(),
            "score": None,
            "topic_id": sig.topic_id,
            "href": f"/module/{sig.topic_id}" if sig.topic_id else None,
        })

    # 4. Self-assessment & material rating
    feedback_rows = (
        await db.execute(
            select(DBProgressSignal)
            .where(DBProgressSignal.session_id == session_uuid)
            .where(
                DBProgressSignal.signal_type.in_(
                    ["self_assessment", "material_rating"]
                )
            )
            .order_by(DBProgressSignal.created_at.desc())
            .limit(per_source)
        )
    ).scalars().all()
    for sig in feedback_rows:
        if not sig.created_at:
            continue
        title = (
            topics.get(sig.topic_id).title
            if sig.topic_id and sig.topic_id in topics
            else "Topik"
        )
        val = float(sig.value)
        # Values may be 0-1 or 1-5
        if val <= 1:
            display = f"{round(val * 100)}%"
            score_out = round(val * 100)
        else:
            display = f"{val:g}/5"
            score_out = round((val / 5) * 100)
        label = (
            "Penilaian diri"
            if sig.signal_type == "self_assessment"
            else "Rating materi"
        )
        events.append({
            "id": f"sig-{sig.id}",
            "type": "assessment",
            "title": f"{label}: {title}",
            "description": display,
            "created_at": sig.created_at.isoformat(),
            "score": score_out,
            "topic_id": sig.topic_id,
            "href": f"/module/{sig.topic_id}" if sig.topic_id else None,
        })

    # 5. User chat messages (skip empty / system noise)
    chat_rows = (
        await db.execute(
            select(DBChatMessage)
            .where(DBChatMessage.session_id == session_uuid)
            .where(DBChatMessage.role == "user")
            .order_by(DBChatMessage.created_at.desc())
            .limit(per_source)
        )
    ).scalars().all()
    for msg in chat_rows:
        if not msg.created_at or not (msg.content or "").strip():
            continue
        snippet = (msg.content or "").strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "…"
        topic_label = (
            topics.get(msg.topic_id).title
            if msg.topic_id and msg.topic_id in topics
            else None
        )
        events.append({
            "id": f"chat-{msg.id}",
            "type": "chat",
            "title": f"Chat{f': {topic_label}' if topic_label else ''}",
            "description": snippet,
            "created_at": msg.created_at.isoformat(),
            "score": None,
            "topic_id": msg.topic_id,
            "href": f"/chat/{msg.topic_id}" if msg.topic_id else "/chat",
        })

    events.sort(key=lambda e: e["created_at"] or "", reverse=True)
    return events[:limit]
