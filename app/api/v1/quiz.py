import uuid
import random
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.dependencies import get_current_user, verify_session_owner, verify_topic_owner
from app.models.user import User
from app.models.agent import QuizResult, ProgressSignal as DBProgressSignal
from app.models.learning import Topic
from app.schemas.quiz import QuizResponse, QuizSubmission, QuizResultResponse, QuizQuestion
from app.services.learning_service import LearningService
from app.services.quiz_cache import (
    store_quiz,
    consume_quiz,
    get_quiz as get_cached_quiz,
)
from app.tasks.generate_module import generate_module_for_topic
from app.agents.tutor import tutor_generate_quiz

router = APIRouter()


@router.get("/{topic_id}", response_model=QuizResponse)
async def get_quiz(
    topic_id: str,
    num_questions: int = 5,
    time_limit_per_question: int = 60,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate MCQ quiz for a topic and cache it in Redis (30 min TTL).

    Returns a quiz_id alongside the questions; clients must send the
    quiz_id back on submit so we can grade against the exact same
    questions (replay-safe, token-efficient).

    The topic must belong to a session owned by the caller — otherwise any
    authenticated user could generate (and burn LLM tokens on) quizzes for
    another user's topics.
    """
    # Ownership: topic must belong to one of the caller's sessions.
    topic = await verify_topic_owner(topic_id, current_user, db)
    session = await verify_session_owner(topic.session_id, current_user, db)

    # Cooldown Logic: 10 mins for first fail, 5 mins for subsequent fails
    from sqlalchemy import select
    from datetime import datetime, timezone
    quiz_results = await db.execute(
        select(QuizResult)
        .where(QuizResult.session_id == session.id, QuizResult.topic_id == topic_id)
        .order_by(QuizResult.created_at.desc())
    )
    results = quiz_results.scalars().all()
    failed_results = [r for r in results if r.score < 0.8]
    
    if failed_results and results[0].score < 0.8:
        # The most recent attempt was a failure
        fail_count = len(failed_results)
        cooldown_mins = 10 if fail_count == 1 else 5
        cooldown_seconds = cooldown_mins * 60
        
        # Calculate time passed since the last failure
        last_failure_time = results[0].created_at
        if last_failure_time.tzinfo is None:
            last_failure_time = last_failure_time.replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        elapsed = (now - last_failure_time).total_seconds()
        
        if elapsed < cooldown_seconds:
            remaining = int(cooldown_seconds - elapsed)
            raise HTTPException(
                status_code=429,
                detail={"message": "cooldown", "remaining_seconds": remaining}
            )

    # Dynamic difficulty and question count based on mastery_score
    mastery = topic.mastery_score or 0.0
    
    # If the client explicitly sends a num_questions, we still respect it (or we can override).
    # Since we want it fully dynamic for the quiz flow, we'll calculate it if num_questions <= 5 (default).
    if mastery < 0.4:
        dynamic_num = 10
        difficulty = "mudah"
    elif mastery < 0.7:
        dynamic_num = 15
        difficulty = "menengah"
    else:
        dynamic_num = 20
        difficulty = "sulit"

    questions_data = await tutor_generate_quiz(
        user_id=str(current_user.id),
        topic_id=topic_id,
        topic_title=topic_id.replace("_", " ").title(),
        language=session.language,
        num_questions=dynamic_num,
        difficulty=difficulty,
    )

    if questions_data:
        random.shuffle(questions_data)

    if not questions_data:
        raise HTTPException(
            status_code=404,
            detail="Could not generate quiz. Make sure the topic has been indexed first.",
        )

    # Cache the questions and obtain a stable quiz_id
    quiz_id = await store_quiz(
        user_id=str(current_user.id),
        topic_id=topic_id,
        questions=questions_data,
    )

    questions = [QuizQuestion(**q) for q in questions_data]
    time_limit_seconds = time_limit_per_question * len(questions)
    return QuizResponse(
        quiz_id=quiz_id,
        topic_id=topic_id,
        questions=questions,
        total_questions=len(questions),
        time_limit_seconds=time_limit_seconds,
    )


@router.post("/submit", response_model=QuizResultResponse)
async def submit_quiz(
    submission: QuizSubmission,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit quiz answers and calculate score.

    Grading source priority:
    1. The Redis cache (looked up by quiz_id, then by (user, topic)).
    2. submission.questions_data (back-compat for clients that send it).
    3. Fall back to re-generating via LLM (logged as a warning — only
       happens if the cache expired or the client is missing quiz_id).

    The submitted (session_id, topic_id) pair must belong to the caller.
    The session ownership and the topic↔session scoping are both verified
    before grading, so a result cannot be persisted against another user's
    session/topic.
    """
    session = await verify_session_owner(submission.session_id, current_user, db)
    await verify_topic_owner(
        submission.topic_id, current_user, db, require_session_id=session.id
    )

    quiz_data: list[dict] | None = None
    graded_quiz_id: str | None = None
    cache_hit = False

    # 1. Try the Redis cache first
    if submission.quiz_id:
        cached = await consume_quiz(
            user_id=str(current_user.id),
            topic_id=submission.topic_id,
            quiz_id=submission.quiz_id,
        )
        if cached:
            quiz_data = cached.get("questions")
            graded_quiz_id = cached.get("quiz_id")
            cache_hit = True

    if quiz_data is None:
        # 2. Side-index fallback (quiz_id absent but quiz still in cache)
        cached = await get_cached_quiz(
            user_id=str(current_user.id), topic_id=submission.topic_id
        )
        if cached:
            quiz_data = cached.get("questions")
            graded_quiz_id = cached.get("quiz_id")
            cache_hit = True

    if quiz_data is None and submission.questions_data:
        # 3a. Legacy fallback — client sent questions_data
        quiz_data = submission.questions_data

    if quiz_data is None:
        # 3b. No quiz found in cache AND no questions_data sent.
        # We intentionally do NOT re-generate via LLM anymore — that
        # path was token-expensive and produced inconsistent scores
        # (LLM non-determinism). The client should always include
        # quiz_id (which the frontend does) or questions_data.
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"[QUIZ_SUBMIT] No cached quiz for user={current_user.id} "
            f"topic={submission.topic_id} quiz_id={submission.quiz_id}. "
            f"Client may be reusing expired or reloaded page."
        )
        raise HTTPException(
            status_code=410,
            detail="Quiz sudah kedaluwarsa. Silakan mulai kuis baru.",
        )

    if not quiz_data:
        raise HTTPException(status_code=400, detail="Quiz data not available")

    # Calculate score
    correct = 0
    answers_detail = []
    for i, answer in enumerate(submission.answers):
        if i < len(quiz_data):
            q = quiz_data[i]
            is_correct = answer.selected_answer == q.get("correct_answer", "")
            if is_correct:
                correct += 1
            answers_detail.append({
                "question_index": i,
                "selected": answer.selected_answer,
                "correct": q.get("correct_answer", ""),
                "is_correct": is_correct,
            })

    total = len(quiz_data)
    score = correct / total if total > 0 else 0.0
    percentage = round(score * 100, 1)

    # Persist result
    service = LearningService(db)
    quiz_result = QuizResult(
        session_id=submission.session_id,
        topic_id=submission.topic_id,
        score=score,
        total_questions=total,
        correct_answers=correct,
        answers_detail=answers_detail,
        time_spent_seconds=submission.time_spent_seconds,
    )
    db.add(quiz_result)
    await db.commit()

    # Insert ProgressSignal for quiz_score
    new_signal = DBProgressSignal(
        id=uuid.uuid4(),
        session_id=submission.session_id,
        topic_id=submission.topic_id,
        signal_type="quiz_score",
        value=score
    )
    db.add(new_signal)
    await db.commit()

    import asyncio
    asyncio.create_task(
        _run_post_quiz_evaluation(
            str(submission.session_id), 
            str(current_user.id), 
            submission.topic_id
        )
    )

    # Generate feedback
    if percentage >= 90:
        feedback = "Luar biasa! Anda telah menguasai materi ini dengan sangat baik."
    elif percentage >= 80:
        feedback = "Bagus! Anda memahami sebagian besar materi. Tinjau kembali bagian yang masih salah."
    else:
        feedback = "Perlu lebih banyak latihan. Coba baca ulang materi dan diskusikan dengan Tutor AI."

    cooldown_remaining_seconds = 0
    if score < 0.8:
        from sqlalchemy import select
        quiz_results = await db.execute(
            select(QuizResult)
            .where(QuizResult.session_id == submission.session_id, QuizResult.topic_id == submission.topic_id)
        )
        results = quiz_results.scalars().all()
        failed_count = sum(1 for r in results if r.score < 0.8)
        # Because we just inserted the current one, failed_count includes it.
        cooldown_mins = 10 if failed_count == 1 else 5
        cooldown_remaining_seconds = cooldown_mins * 60

    from sqlalchemy import select
    from app.models.agent import ProgressSignal
    from app.agents.feedback_engine import calculate_mastery, evaluate_mastery
    from app.agents.state import ProgressSignals

    signal_result = await db.execute(
        select(ProgressSignal)
        .where(ProgressSignal.session_id == submission.session_id)
        .where(ProgressSignal.topic_id == submission.topic_id)
        .order_by(ProgressSignal.created_at.asc())
    )
    db_signals = signal_result.scalars().all()
    agg_signals = {s.signal_type: s.value for s in db_signals}
    agg_signals["quiz_score"] = score
    
    signals = ProgressSignals(
        quiz_score=agg_signals.get("quiz_score"),
        reading_time_ratio=agg_signals.get("reading_time_ratio"),
        question_frequency=agg_signals.get("question_frequency"),
        self_assessment=agg_signals.get("self_assessment"),
        material_rating=agg_signals.get("material_rating")
    )
    mastery = calculate_mastery(signals)
    feedback_action_val = evaluate_mastery(mastery, submission.topic_id).action

    return QuizResultResponse(
        score=score,
        total_questions=total,
        correct_answers=correct,
        percentage=percentage,
        feedback=feedback,
        cooldown_remaining_seconds=cooldown_remaining_seconds,
        feedback_action=feedback_action_val,
    )


async def trigger_next_module_after_quiz(session_id: uuid.UUID, user_id: str, topic_id: str, db: AsyncSession):
    """
    Mark the passed topic as completed, activate the next topic, and
    kick off module generation for that next topic.

    Passing a quiz (score >= 0.80) is the source of truth for completion —
    the current topic must flip to ``completed`` even if the user never
    clicks "Lanjut Topik Berikutnya" on the quiz result screen.
    """
    import logging
    from datetime import datetime, timezone

    logger = logging.getLogger(__name__)

    service = LearningService(db)

    # 1) Complete the topic the user just passed.
    current = await service.get_topic(topic_id)
    if current is None:
        logger.warning(f"[EVALUATION] Topic {topic_id} not found while completing after quiz")
        return

    if current.status != "completed":
        current.status = "completed"
        if current.completed_at is None:
            current.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(current)
        logger.info(f"[EVALUATION] Marked topic {topic_id} as completed after quiz pass")
    else:
        logger.info(f"[EVALUATION] Topic {topic_id} already completed")

    # 2) Unlock / activate the next curriculum topic.
    next_topic = await service.activate_next_topic(session_id, topic_id)

    if next_topic:
        logger.info(f"[EVALUATION] Triggering module generation for next topic: {next_topic.id}")
        generate_module_for_topic.delay(
            session_id=str(session_id),
            user_id=user_id,
            topic_id=next_topic.id,
        )
    else:
        logger.info(f"[EVALUATION] No next topic to generate module for after {topic_id}")


async def _run_post_quiz_evaluation(session_id: str, user_id: str, topic_id: str) -> None:
    """
    Phase polish #4 — fire-and-forget feedback evaluation after a quiz is submitted.
    Reads accumulated signals, computes mastery, persists a FeedbackAction, and 
    triggers a curriculum replan or continues to the next topic.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        from app.config import settings
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import select
        from app.db.database import async_sessionmaker
        from app.models.learning import Topic
        from app.models.agent import ProgressSignal
        from app.agents.feedback_engine import calculate_mastery, evaluate_mastery
        from app.agents.state import ProgressSignals
        import uuid
        from datetime import datetime

        engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
        SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

        async with SessionLocal() as db:
            topic = (await db.execute(
                select(Topic).where(Topic.id == topic_id)
            )).scalar_one_or_none()
            if not topic:
                logger.warning(f"[POST-QUIZ-EVAL] Topic {topic_id} not found")
                return

            # Fetch all signals, ordered by created_at asc so newest overwrites oldest
            signal_result = await db.execute(
                select(ProgressSignal)
                .where(ProgressSignal.session_id == uuid.UUID(session_id))
                .where(ProgressSignal.topic_id == topic_id)
                .order_by(ProgressSignal.created_at.asc())
            )
            db_signals = signal_result.scalars().all()

            agg_signals = {}
            for s in db_signals:
                agg_signals[s.signal_type] = s.value

            signals = ProgressSignals(
                quiz_score=agg_signals.get("quiz_score"),
                reading_time_ratio=agg_signals.get("reading_time_ratio"),
                question_frequency=agg_signals.get("question_frequency"),
                self_assessment=agg_signals.get("self_assessment"),
                material_rating=agg_signals.get("material_rating")
            )

            score = calculate_mastery(signals)
            action = evaluate_mastery(score, topic_id)
            logger.info(f"[POST-QUIZ-EVAL] {topic_id} mastery={score:.2f} -> {action.action}")

            # Update topic mastery + award XP for any newly-crossed
            # milestones. This is the SINGLE hook that connects the
            # 5-signal mastery system to the XP/level gamification —
            # every signal update eventually flows through here (or
            # through similar calls if other endpoints also recompute
            # mastery), so we get full coverage without touching chat
            # or module endpoints.
            old_mastery = topic.mastery_score or 0.0
            topic.mastery_score = score
            topic.quiz_score = signals.quiz_score
            topic.reading_time_ratio = signals.reading_time_ratio
            topic.question_frequency_score = signals.question_frequency
            topic.self_assessment_score = signals.self_assessment
            topic.material_rating_score = signals.material_rating
            topic.feedback_action = action.action

            # XP awards — see xp_service.award_mastery_milestone_xp.
            # Returns a list of awarded events (empty if no new milestones
            # were crossed or if all crossed milestones were already
            # earned). The caller (this background task) commits
            # alongside the mastery update.
            # NOTE: this background task receives the user_id (str)
            # not a User ORM object, so we fetch the user row here
            # to mutate ``total_xp`` in place.
            from app.models.user import User as _User
            from app.services.xp_service import award_mastery_milestone_xp
            _user = (
                await db.execute(
                    select(_User).where(_User.id == uuid.UUID(user_id))
                )
            ).scalar_one_or_none()
            xp_awards: list = []
            if _user is not None:
                xp_awards = await award_mastery_milestone_xp(
                    db=db,
                    user=_user,
                    topic_id=topic_id,
                    old_mastery=old_mastery,
                    new_mastery=score,
                )
            if xp_awards:
                logger.info(
                    "[POST-QUIZ-EVAL] user=%s topic=%s xp_awards=%s",
                    user_id, topic_id, xp_awards,
                )
                
                # Check for level up
                leveled_up_events = [a for a in xp_awards if a.get("leveled_up")]
                if leveled_up_events:
                    new_level = leveled_up_events[-1].get("new_level")
                    from app.services.notification_service import NotificationService
                    notif_service = NotificationService(db)
                    await notif_service.create_notification(
                        user_id=user_id,
                        title=f"Naik Level: Level {new_level}!",
                        message=f"Selamat! Anda berhasil mencapai Level {new_level}.",
                        notification_type="level_up",
                        link="/dashboard"
                    )

            # Execute supplementary generation if needed (remedial or enrichment)
            if action.action in ("remedial", "enrichment"):
                # Check if module already has the supplementary material
                from app.models.learning import LearningModule as DBLearningModule
                existing_module = (
                    await db.execute(
                        select(DBLearningModule)
                        .where(DBLearningModule.topic_id == topic_id)
                    )
                ).scalar_one_or_none()
                
                needs_generation = True
                if existing_module:
                    if action.action == "remedial" and existing_module.remedial_markdown:
                        needs_generation = False
                        logger.info(f"[POST-QUIZ-EVAL] Remedial module already exists for {topic_id}, skipping generation.")
                    elif action.action == "enrichment" and existing_module.deep_dive_markdown:
                        needs_generation = False
                        logger.info(f"[POST-QUIZ-EVAL] Enrichment (Deep Dive) module already exists for {topic_id}, skipping generation.")
                        
                if needs_generation:
                    logger.info(f"[POST-QUIZ-EVAL] Action is {action.action}, triggering supplementary module generation...")
                    
                    context_str = ""
                    if action.action == "remedial":
                        from app.models.agent import QuizResult as DBQuizResult
                        quiz_res = await db.execute(
                            select(DBQuizResult)
                            .where(DBQuizResult.session_id == uuid.UUID(session_id))
                            .where(DBQuizResult.topic_id == topic_id)
                            .order_by(DBQuizResult.created_at.desc())
                        )
                        latest_quiz = quiz_res.scalars().first()
                        
                        wrong_context = []
                        if latest_quiz and latest_quiz.answers_detail:
                            for detail in latest_quiz.answers_detail:
                                if not detail.get("is_correct"):
                                    q_text = detail.get("question_text", f"Topik soal ke-{detail.get('question_index', 0)+1}")
                                    wrong_context.append(f"Q: {q_text}")
                        
                        if wrong_context:
                            context_str = "Topik remedial harus difokuskan pada perbaikan pemahaman pertanyaan berikut: " + "; ".join(wrong_context)

                    from app.tasks.generate_supplementary import generate_supplementary_module
                    generate_supplementary_module.delay(
                        session_id=session_id,
                        user_id=user_id,
                        topic_id=topic_id,
                        supplementary_type="deep_dive" if action.action == "enrichment" else "remedial",
                        context=context_str
                    )

            await db.commit()

            # Lulus topik ditentukan OLEH quiz_score >= 0.80
            quiz_passed = (signals.quiz_score is not None and signals.quiz_score >= 0.80)
            
            if quiz_passed:
                logger.info(f"[POST-QUIZ-EVAL] Quiz passed (>= 0.80). Triggering next module after {topic_id}...")
                await trigger_next_module_after_quiz(uuid.UUID(session_id), user_id, topic_id, db)
            else:
                logger.info(f"[POST-QUIZ-EVAL] Quiz score {signals.quiz_score} < 0.80, holding off on next module.")
                
            await engine.dispose()
    except Exception as e:
        logger.error(f"[POST-QUIZ-EVAL] Background failed: {e}")


@router.get("/history/{session_id}")
async def get_quiz_history(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all quiz attempts for a learning session, joined with topic
    title for display in the UI.

    Each attempt is enriched with:
    - ``topic_title`` (joined from `Topic`) — so the frontend can
      render "Kuis: <topic title>" without a separate topics query.
    - ``time_spent_seconds`` — total time the user spent on the
      attempt (for stats like "5 menit per kuis").
    - ``attempt_number`` — the Nth attempt at this topic (1-indexed).
      Useful for the per-topic history page to show "Attempt 1 of 3".

    The list is sorted by ``created_at DESC`` so the most recent
    attempt is first.
    """
    # Ownership: the session must belong to the caller before we expose
    # its quiz history. FastAPI has already coerced the path param to a
    # UUID; verify_session_owner accepts a UUID directly.
    await verify_session_owner(session_id, current_user, db)

    from sqlalchemy import select
    result = await db.execute(
        select(QuizResult)
        .where(QuizResult.session_id == session_id)
        # Join Topic to get the title (avoids N+1 per-row queries).
        .outerjoin(Topic, QuizResult.topic_id == Topic.id)
        .order_by(QuizResult.created_at.desc())
    )
    # We re-query the distinct topic_ids in this result and zip
    # titles by id — clean and avoids any lazy-load surprises.
    records = result.scalars().all()
    topic_ids = list({r.topic_id for r in records if r.topic_id})
    topic_titles: dict[str, str] = {}
    if topic_ids:
        topics_result = await db.execute(
            select(Topic.id, Topic.title).where(Topic.id.in_(topic_ids))
        )
        topic_titles = {row[0]: row[1] for row in topics_result.all()}

    return [
        {
            "id": str(r.id),
            "topic_id": r.topic_id,
            "topic_title": topic_titles.get(r.topic_id) if r.topic_id else None,
            "score": r.score,
            "total_questions": r.total_questions,
            "correct_answers": r.correct_answers,
            **_build_attempt_extras(r),
            "created_at": r.created_at.isoformat() if r.created_at is not None else None,
        }
        for r in records
    ]


def _build_attempt_extras(quiz_result) -> dict:
    """
    Compute percentage + iso created_at from a QuizResult row.

    Pulled out of the dict-comprehension above so Pyright can resolve
    the SQLAlchemy column types cleanly (inside the comprehension,
    ``r.score`` is typed as ``Column[Unknown]`` even though at
    runtime it's a ``float``).
    """
    score_value = float(quiz_result.score) if quiz_result.score is not None else 0.0
    return {
        "percentage": round(score_value * 100, 1),
        "time_spent_seconds": quiz_result.time_spent_seconds,
        "attempt_number": quiz_result.attempt_number,
    }


@router.get("/history/{session_id}/topic/{topic_id}")
async def get_quiz_history_by_topic(
    session_id: uuid.UUID,
    topic_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all quiz attempts for ONE topic within a session, ordered
    by ``created_at ASC`` (oldest first) — so the frontend can
    show attempt #1, #2, #3 in chronological order on the
    per-topic history page.

    Returns the same shape as ``get_quiz_history`` but filtered
    to a single topic. The frontend uses this to render the
    "Attempt N of M" progression chart per topic.
    """
    # Ownership: both the session and the (session-scoped) topic must
    # belong to the caller.
    await verify_session_owner(session_id, current_user, db)
    await verify_topic_owner(topic_id, current_user, db, require_session_id=session_id)

    from sqlalchemy import select
    result = await db.execute(
        select(QuizResult)
        .where(QuizResult.session_id == session_id)
        .where(QuizResult.topic_id == topic_id)
        .order_by(QuizResult.created_at.asc())
    )
    records = result.scalars().all()

    # Topic title (single row, but join cleanly).
    topic_title = None
    if topic_id:
        t = await db.execute(
            select(Topic.title).where(Topic.id == topic_id)
        )
        topic_title = t.scalar_one_or_none()

    return {
        "topic_id": topic_id,
        "topic_title": topic_title,
        "attempts": [
            {
                "id": str(r.id),
                "topic_id": r.topic_id,
                "score": r.score,
                "total_questions": r.total_questions,
                "correct_answers": r.correct_answers,
                **_build_attempt_extras(r),
                "created_at": r.created_at.isoformat() if r.created_at is not None else None,
            }
            for r in records
        ],
    }


@router.get("/attempt/{attempt_id}")
async def get_quiz_attempt_detail(
    attempt_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch a single quiz attempt with the full ``answers_detail``
    payload (which question the user got right/wrong, what they
    selected vs. the correct answer).

    Used by the per-topic history page's "Review" action — shows
    the user which questions they missed so they can go re-read
    the material and retry. Returns 404 if the attempt doesn't
    exist or belongs to a different user.
    """
    from sqlalchemy import select
    from app.models.learning import LearningSession
    result = await db.execute(
        select(QuizResult, LearningSession.user_id)
        .join(LearningSession, QuizResult.session_id == LearningSession.id)
        .where(QuizResult.id == attempt_id)
    )
    row = result.first()
    if row is None:
        raise HTTPException(status_code=404, detail="Quiz attempt not found")
    quiz_result, session_user_id = row
    # Defense-in-depth: ensure the requester owns this session.
    if str(session_user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Forbidden")

    # Look up the topic title (best-effort — may be null if the
    # Topic row was deleted but the QuizResult persists).
    topic_title = None
    if quiz_result.topic_id:
        t = await db.execute(
            select(Topic.title).where(Topic.id == quiz_result.topic_id)
        )
        topic_title = t.scalar_one_or_none()

    # Compute the percentage from the score (score is 0-1).
    score_value = float(quiz_result.score) if quiz_result.score is not None else 0.0
    percentage = round(score_value * 100, 1)
    created_at_iso = (
        quiz_result.created_at.isoformat() if quiz_result.created_at is not None else None
    )

    return {
        "id": str(quiz_result.id),
        "session_id": str(quiz_result.session_id),
        "topic_id": quiz_result.topic_id,
        "topic_title": topic_title,
        "score": score_value,
        "total_questions": quiz_result.total_questions,
        "correct_answers": quiz_result.correct_answers,
        "percentage": percentage,
        "time_spent_seconds": quiz_result.time_spent_seconds,
        "attempt_number": quiz_result.attempt_number,
        "answers_detail": quiz_result.answers_detail or [],
        "created_at": created_at_iso,
    }
