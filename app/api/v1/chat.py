import uuid
import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db, async_sessionmaker
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse, ChatHistoryMessage, ChatSource, RAGMetrics, RAGASSummary
from app.services.learning_service import LearningService
from app.agents.tutor import tutor_chat
from app.rag.indexer import extract_text_from_file, index_uploaded_document
from app.rag.evaluator import get_rag_evaluator
from app.models.agent import ChatMessage
logger = logging.getLogger(__name__)

router = APIRouter()


async def _run_ragas_background(
    db_url: str,
    message_id: str,
    question: str,
    answer: str,
    contexts: list[str],
) -> None:
    """
    Background task: score a chat response with RAGAS (or fallback)
    and update the chat_messages row. Never raises.
    """
    try:
        evaluator = get_rag_evaluator()
        if evaluator is None:
            return
        scores = await evaluator.evaluate(question, answer, contexts)
        if scores.get("rag_faithfulness") is None and scores.get("rag_answer_relevancy") is None:
            return
        # Use a fresh session (background task can't share request session)
        from app.config import settings
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
        SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
        async with SessionLocal() as db:
            from sqlalchemy import update
            await db.execute(
                update(ChatMessage)
                .where(ChatMessage.id == uuid.UUID(message_id))
                .values(
                    rag_faithfulness=scores.get("rag_faithfulness"),
                    rag_answer_relevancy=scores.get("rag_answer_relevancy"),
                )
            )
            await db.commit()
        logger.info(
            f"[RAGAS] Scored message {message_id[:8]}... "
            f"faith={scores.get('rag_faithfulness'):.2f} "
            f"rel={scores.get('rag_answer_relevancy'):.2f} "
            f"method={scores.get('method')}"
        )
    except Exception as e:
        logger.error(f"[RAGAS] Background eval failed: {e}")

@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a message to the Tutor Agent and receive a RAG-based response."""
    service = LearningService(db)

    # Verify the session belongs to the current user
    session = await service.get_session(request.session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    # Get recent chat history for context
    history_msgs = await service.get_chat_history(request.session_id, request.topic_id)
    chat_history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_msgs
    ]

    # Save user message
    await service.save_chat_message(
        session_id=request.session_id,
        topic_id=request.topic_id,
        role="user",
        content=request.message,
    )

    # Call Tutor Agent
    result = await tutor_chat(
        user_id=str(current_user.id),
        session_id=str(request.session_id),
        topic_id=request.topic_id,
        query=request.message,
        chat_history=chat_history,
        include_sources=request.include_sources,
    )

    # Save assistant response
    saved_msg = await service.save_chat_message(
        session_id=request.session_id,
        topic_id=request.topic_id,
        role="assistant",
        content=result["response"],
        sources=result.get("sources", []),
        latency_ms=result.get("latency_ms"),
    )

    # ── Fire-and-forget RAGAS evaluation (Phase polish #1) ──
    # Don't block the user response. If RAGAS isn't installed or the
    # LLM judge fails, we still return the response successfully.
    chunks = result.get("chunks", [])
    contexts = [c.get("text", "") for c in chunks] if chunks else []
    if not contexts and result.get("sources"):
        contexts = [s.get("title", "") for s in result.get("sources", [])]

    asyncio.create_task(
        _run_ragas_background(
            db_url=str(request.session_id),
            message_id=str(saved_msg.id),
            question=request.message,
            answer=result["response"],
            contexts=contexts,
        )
    )

    return ChatResponse(
        message_id=str(saved_msg.id),
        response=result["response"],
        sources=[
            ChatSource(**s) for s in result.get("sources", [])
        ],
        rag_metrics=RAGMetrics(
            latency_ms=result.get("latency_ms", 0),
            chunks_used=result.get("chunks_used", 0),
        ),
    )


@router.get("/history/{topic_id}", response_model=list[ChatHistoryMessage])
async def get_chat_history(
    topic_id: str,
    session_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get chat history for a specific topic."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    messages = await service.get_chat_history(session_id, topic_id, limit, offset)
    return [
        ChatHistoryMessage(
            id=str(msg.id),
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at.isoformat() if msg.created_at else "",
            rag_faithfulness=msg.rag_faithfulness,
            rag_answer_relevancy=msg.rag_answer_relevancy,
        )
        for msg in messages
    ]


@router.get("/ragas-summary/{session_id}", response_model=RAGASSummary)
async def get_ragas_summary(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate RAGAS scores across all chat messages in a session.
    Used by the dashboard widget to show the user the average quality
    of Tutor RAG responses over time.
    """
    from sqlalchemy import select, func
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    # Aggregate
    from sqlalchemy import case
    stmt = select(
        func.count(ChatMessage.id).label("total"),
        func.avg(ChatMessage.rag_faithfulness).label("avg_faith"),
        func.avg(ChatMessage.rag_answer_relevancy).label("avg_rel"),
        # Count rows where EITHER score is < 0.5 (flagged)
        func.coalesce(
            func.sum(
                case(
                    (
                        (ChatMessage.rag_faithfulness < 0.5)
                        | (ChatMessage.rag_answer_relevancy < 0.5),
                        1,
                    ),
                    else_=0,
                )
            ),
            0,
        ).label("flagged"),
    ).where(
        ChatMessage.session_id == session_id,
        ChatMessage.role == "assistant",
    )
    row = (await db.execute(stmt)).one()

    # Count scored (non-null)
    scored_stmt = select(func.count(ChatMessage.id)).where(
        ChatMessage.session_id == session_id,
        ChatMessage.role == "assistant",
        ChatMessage.rag_faithfulness.is_not(None),
    )
    scored_count = (await db.execute(scored_stmt)).scalar_one()

    # p95 — use SQL percentile
    from sqlalchemy import text
    p95_stmt = text("""
        SELECT
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY rag_faithfulness) AS p95_faith,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY rag_answer_relevancy) AS p95_rel
        FROM chat_messages
        WHERE session_id = :sid
          AND role = 'assistant'
          AND rag_faithfulness IS NOT NULL
    """).bindparams(sid=session_id)
    p95_row = (await db.execute(p95_stmt)).one()

    return RAGASSummary(
        session_id=str(session_id),
        total_messages=int(row.total or 0),
        scored_messages=int(scored_count or 0),
        avg_faithfulness=float(row.avg_faith) if row.avg_faith is not None else None,
        avg_answer_relevancy=float(row.avg_rel) if row.avg_rel is not None else None,
        p95_faithfulness=float(p95_row.p95_faith) if p95_row.p95_faith is not None else None,
        p95_answer_relevancy=float(p95_row.p95_rel) if p95_row.p95_rel is not None else None,
        flagged_messages=int(row.flagged or 0),
    )


@router.delete("/history/{topic_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_history(
    topic_id: str,
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all chat history for a specific topic."""
    from sqlalchemy import delete
    from app.models.agent import ChatMessage
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    await db.execute(
        delete(ChatMessage).where(
            ChatMessage.session_id == session_id,
            ChatMessage.topic_id == topic_id,
        )
    )
    await db.commit()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    topic_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document to be indexed for RAG within a specific topic."""
    service = LearningService(db)

    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await service.get_session(session_uuid)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    # Read file
    file_bytes = await file.read()
    
    # 10 MB limit check
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    try:
        # Extract text
        raw_text = extract_text_from_file(file_bytes, file.filename)
        if not raw_text:
            raise HTTPException(status_code=400, detail="No readable text found in document.")

        # Index document
        chunks_indexed = index_uploaded_document(
            user_id=str(current_user.id),
            session_id=str(session_id),
            topic_id=str(topic_id),
            filename=file.filename,
            raw_text=raw_text,
        )

        return {
            "status": "success",
            "message": f"Document '{file.filename}' indexed successfully.",
            "chunks": chunks_indexed,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


