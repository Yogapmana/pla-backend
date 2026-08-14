import uuid
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.chat import (
    ChatRequest, ChatResponse, ChatHistoryMessage, ChatSource, RAGMetrics,
    RAGASSummary, ChatSessionCreate, ChatSessionResponse,
    ChatJobAccepted, ChatJobStatus,
)
from app.services.learning_service import LearningService
from app.tasks.chat_tasks import tutor_chat_task, general_chatbot_task
from app.utils.redis_client import get_redis
from app.rag.indexer import extract_text_from_file, index_uploaded_document
from app.models.agent import ChatMessage
from app.models.learning import LearningSession
import re

logger = logging.getLogger(__name__)

def strip_hidden_content(text: str) -> str:
    """Remove the hidden document content block from the message before returning to the UI."""
    return re.sub(r'<!-- UPLOADED_DOCUMENT_CONTENT:.*?-->', '', text, flags=re.DOTALL).strip()

router = APIRouter()

@router.get("/sessions", response_model=list[ChatSessionResponse])
async def get_chat_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = LearningService(db)
    sessions = await service.get_chat_sessions(current_user.id)
    return [
        ChatSessionResponse(
            id=str(s.id),
            topic=s.topic,
            created_at=s.created_at.isoformat() if s.created_at else ""
        ) for s in sessions
    ]

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    request: ChatSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = LearningService(db)
    session = await service.create_session(
        user_id=current_user.id,
        topic=request.title,
        level="General Chat",
        duration_weeks=0,
        hours_per_day=0.0
    )
    return ChatSessionResponse(
        id=str(session.id),
        topic=session.topic,
        created_at=session.created_at.isoformat() if session.created_at else ""
    )

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = LearningService(db)
    session = await service.get_session(uuid.UUID(session_id))
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    await service.delete_session(uuid.UUID(session_id))
    return {"status": "success"}

@router.post("/message", status_code=202, response_model=ChatJobAccepted)
async def send_message(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a chat message and enqueue the LLM work to Celery.

    Returns 202 + job_id immediately; the assistant reply is generated in a
    Celery task and persisted. Poll GET /chat/job/{job_id} for completion.
    """
    try:
        service = LearningService(db)

        # Verify the session belongs to the current user
        from app.dependencies import verify_session_owner, verify_topic_owner

        session = await verify_session_owner(request.session_id, current_user, db)
        if request.topic_id:
            await verify_topic_owner(
                request.topic_id, current_user, db, require_session_id=session.id
            )

        # Get recent chat history for context
        history_msgs = await service.get_chat_history(request.session_id, request.topic_id)
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in history_msgs
        ]

        # Save user message immediately so the UI shows it without waiting
        await service.save_chat_message(
            session_id=request.session_id,
            topic_id=request.topic_id,
            role="user",
            content=request.message,
        )

        # Enqueue the LLM work off the request path.
        if request.topic_id:
            task = tutor_chat_task.delay(
                user_id=str(current_user.id),
                session_id=str(request.session_id),
                topic_id=request.topic_id,
                query=request.message,
                language=session.language,
                chat_history=chat_history,
                include_sources=request.include_sources,
            )
        else:
            task = general_chatbot_task.delay(
                user_id=str(current_user.id),
                session_id=str(request.session_id),
                query=request.message,
                chat_history=chat_history,
            )
        return ChatJobAccepted(job_id=task.id)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        logger.error(f"Error in chat endpoint: {err_msg}")
        raise HTTPException(status_code=400, detail=str(err_msg))


@router.get("/job/{job_id}", response_model=ChatJobStatus)
async def get_chat_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Poll the status of a chat job (Celery task id from POST /chat/message).

    Status is one of: queued | running | done | failed. 404 when the job
    record has expired (TTL 15 min) — the client should refetch chat history.
    """
    r = await get_redis()
    raw = await r.get(f"chat_job:{job_id}")
    if not raw:
        raise HTTPException(
            status_code=404,
            detail="Chat job tidak ditemukan atau sudah kedaluwarsa.",
        )
    data = json.loads(raw)
    return ChatJobStatus(job_id=job_id, **data)


@router.get("/history", response_model=list[ChatHistoryMessage])
async def get_chat_history(
    session_id: uuid.UUID,
    topic_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get chat history for a specific topic."""
    from app.dependencies import verify_session_owner, verify_topic_owner

    service = LearningService(db)
    session = await verify_session_owner(session_id, current_user, db)
    if topic_id:
        await verify_topic_owner(
            topic_id, current_user, db, require_session_id=session.id
        )

    messages = await service.get_chat_history(session_id, topic_id, limit, offset)
    return [
        ChatHistoryMessage(
            id=str(msg.id),
            role=msg.role,
            content=strip_hidden_content(msg.content),
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

    Results are cached in Redis (TTL 10 min) — invalidated when a new
    assistant message is scored or history is cleared.
    """
    from sqlalchemy import select, func
    from app.services import metrics_cache

    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    cached = await metrics_cache.get_ragas_summary(str(session_id))
    if cached is not None:
        return RAGASSummary(**cached)

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

    summary = RAGASSummary(
        session_id=str(session_id),
        total_messages=int(row.total or 0),
        scored_messages=int(scored_count or 0),
        avg_faithfulness=float(row.avg_faith) if row.avg_faith is not None else None,
        avg_answer_relevancy=float(row.avg_rel) if row.avg_rel is not None else None,
        p95_faithfulness=float(p95_row.p95_faith) if p95_row.p95_faith is not None else None,
        p95_answer_relevancy=float(p95_row.p95_rel) if p95_row.p95_rel is not None else None,
        flagged_messages=int(row.flagged or 0),
    )
    await metrics_cache.set_ragas_summary(str(session_id), summary.model_dump())
    return summary


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_history(
    session_id: uuid.UUID,
    topic_id: str | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all chat history for a specific topic."""
    from sqlalchemy import delete
    from app.models.agent import ChatMessage
    from app.dependencies import verify_session_owner, verify_topic_owner

    session = await verify_session_owner(session_id, current_user, db)
    if topic_id is not None:
        await verify_topic_owner(
            topic_id, current_user, db, require_session_id=session.id
        )

    query = delete(ChatMessage).where(ChatMessage.session_id == session_id)
    if topic_id is not None:
        query = query.where(ChatMessage.topic_id == topic_id)
    else:
        query = query.where(ChatMessage.topic_id.is_(None))

    await db.execute(query)
    await db.commit()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    topic_id: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document to be indexed for RAG within a specific topic."""
    from app.dependencies import verify_session_owner, verify_topic_owner

    service = LearningService(db)

    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await verify_session_owner(session_uuid, current_user, db)
    if topic_id:
        await verify_topic_owner(
            topic_id, current_user, db, require_session_id=session.id
        )

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
            topic_id=str(topic_id) if topic_id else None,
            filename=file.filename,
            raw_text=raw_text,
        )

        # Save a system message so the LLM and user know a document was uploaded
        # We embed the raw text in an HTML comment so it's passed to the LLM but hidden from the user UI
        safe_text = raw_text[:25000] # truncate to ~25k chars to prevent context overflow
        content_for_db = (
            f"*(Sistem: Dokumen '{file.filename}' berhasil diunggah dan siap digunakan sebagai referensi.)*\n\n"
            f"<!-- UPLOADED_DOCUMENT_CONTENT:\n{safe_text}\n-->"
        )
        
        await service.save_chat_message(
            session_id=session_uuid,
            topic_id=topic_id,
            role="assistant",
            content=content_for_db,
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


