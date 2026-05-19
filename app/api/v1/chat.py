import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse, ChatHistoryMessage, ChatSource, RAGMetrics
from app.services.learning_service import LearningService
from app.agents.tutor import tutor_chat

router = APIRouter()

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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get chat history for a specific topic."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    messages = await service.get_chat_history(session_id, topic_id)
    return [
        ChatHistoryMessage(
            id=str(msg.id),
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at.isoformat() if msg.created_at else "",
        )
        for msg in messages
    ]


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
