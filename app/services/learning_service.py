from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.learning import LearningSession, Topic, LearningModule
from app.models.agent import ChatMessage

class LearningService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_session(self, session_id: UUID) -> LearningSession | None:
        result = await self.db.execute(
            select(LearningSession).where(LearningSession.id == session_id)
        )
        return result.scalars().first()

    async def get_topic(self, topic_id: str) -> Topic | None:
        result = await self.db.execute(
            select(Topic).where(Topic.id == topic_id)
        )
        return result.scalars().first()

    async def get_module(self, topic_id: str) -> LearningModule | None:
        result = await self.db.execute(
            select(LearningModule).where(LearningModule.topic_id == topic_id)
        )
        return result.scalars().first()

    async def save_chat_message(
        self,
        session_id: UUID,
        topic_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        latency_ms: int | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            session_id=session_id,
            topic_id=topic_id,
            role=role,
            content=content,
            sources=sources,
            latency_ms=latency_ms,
        )
        self.db.add(msg)
        await self.db.commit()
        await self.db.refresh(msg)
        return msg

    async def get_chat_history(
        self, session_id: UUID, topic_id: str, limit: int = 20
    ) -> list[ChatMessage]:
        result = await self.db.execute(
            select(ChatMessage)
            .where(
                ChatMessage.session_id == session_id,
                ChatMessage.topic_id == topic_id,
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        return list(reversed(messages))  # oldest first
