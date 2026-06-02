import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.agent import QuizResult
from app.schemas.quiz import QuizResponse, QuizSubmission, QuizResultResponse, QuizQuestion
from app.services.learning_service import LearningService
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
    """Generate or retrieve quiz questions for a topic."""
    questions_data = await tutor_generate_quiz(
        user_id=str(current_user.id),
        topic_id=topic_id,
        topic_title=topic_id.replace("_", " ").title(),
        num_questions=num_questions,
    )

    if not questions_data:
        raise HTTPException(
            status_code=404,
            detail="Could not generate quiz. Make sure the topic has been indexed first."
        )

    questions = [QuizQuestion(**q) for q in questions_data]
    time_limit_seconds = time_limit_per_question * len(questions)
    return QuizResponse(
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
    Note: In production, questions should be retrieved from a cache/session
    using a quiz_id returned from GET /quiz/{topic_id}.
    """
    quiz_data = submission.questions_data if submission.questions_data else await tutor_generate_quiz(
        user_id=str(current_user.id),
        topic_id=submission.topic_id,
        topic_title=submission.topic_id.replace("_", " ").title(),
        num_questions=len(submission.answers),
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

    # Trigger next module generation if quiz score >= 80%
    if percentage >= 80:
        await trigger_next_module_after_quiz(
            session_id=submission.session_id,
            user_id=str(current_user.id),
            topic_id=submission.topic_id,
            db=db,
        )

    # Generate feedback
    if percentage >= 80:
        feedback = "Luar biasa! Anda telah menguasai materi ini dengan sangat baik."
    elif percentage >= 60:
        feedback = "Bagus! Anda memahami sebagian besar materi. Tinjau kembali bagian yang masih salah."
    else:
        feedback = "Perlu lebih banyak latihan. Coba baca ulang materi dan diskusikan dengan Tutor AI."

    return QuizResultResponse(
        score=score,
        total_questions=total,
        correct_answers=correct,
        percentage=percentage,
        feedback=feedback,
    )


async def trigger_next_module_after_quiz(session_id: uuid.UUID, user_id: str, topic_id: str, db: AsyncSession):
    """
    After a quiz is submitted with score >= 80%, trigger module generation for the next topic.
    This replaces the old flow where module was generated when topic was marked complete.
    """
    import logging
    logger = logging.getLogger(__name__)

    service = LearningService(db)
    next_topic = await service.activate_next_topic(session_id, topic_id)

    if next_topic:
        logger.info(f"[QUIZ >=80%] Triggering module generation for next topic: {next_topic.id}")
        generate_module_for_topic.delay(
            session_id=str(session_id),
            user_id=user_id,
            topic_id=next_topic.id,
        )
    else:
        logger.info(f"[QUIZ >=80%] No next topic to generate module for after {topic_id}")


@router.get("/history/{session_id}")
async def get_quiz_history(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get quiz attempt history for a learning session."""
    from sqlalchemy.future import select
    result = await db.execute(
        select(QuizResult)
        .where(QuizResult.session_id == session_id)
        .order_by(QuizResult.created_at.desc())
    )
    records = result.scalars().all()
    return [
        {
            "id": str(r.id),
            "topic_id": r.topic_id,
            "score": r.score,
            "total_questions": r.total_questions,
            "correct_answers": r.correct_answers,
            "percentage": round(r.score * 100, 1),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]
