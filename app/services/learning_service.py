from uuid import UUID
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.learning import LearningSession, Topic, LearningModule, ResourceLink
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
        topic_id: str | None,
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
        self, session_id: UUID, topic_id: str | None, limit: int = 50, offset: int = 0
    ) -> list[ChatMessage]:
        query = select(ChatMessage).where(ChatMessage.session_id == session_id)
        if topic_id is not None:
            query = query.where(ChatMessage.topic_id == topic_id)
        else:
            query = query.where(ChatMessage.topic_id.is_(None))
            
        result = await self.db.execute(
            query

            .order_by(ChatMessage.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        messages = result.scalars().all()
        return list(reversed(messages))  # oldest first

    async def create_session(self, user_id: UUID, topic: str, level: str, duration_weeks: int, hours_per_day: float, language: str = "id") -> LearningSession:
        session = LearningSession(
            user_id=user_id,
            topic=topic,
            level=level,
            duration_weeks=duration_weeks,
            hours_per_day=hours_per_day,
            language=language,
            status="processing",
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_sessions(self, user_id: UUID) -> list[LearningSession]:
        result = await self.db.execute(
            select(LearningSession)
            .where(LearningSession.user_id == user_id)
            .where(LearningSession.level != "General Chat")
            .order_by(LearningSession.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_chat_sessions(self, user_id: UUID) -> list[LearningSession]:
        result = await self.db.execute(
            select(LearningSession)
            .where(LearningSession.user_id == user_id)
            .where(LearningSession.level == "General Chat")
            .order_by(LearningSession.created_at.desc())
        )
        return list(result.scalars().all())

    async def delete_session(self, session_id: UUID) -> bool:
        session = await self.get_session(session_id)
        if not session:
            return False
        await self.db.delete(session)
        await self.db.commit()
        return True

    async def save_curriculum(self, session_id: UUID, curriculum_json: dict, version: int = 1) -> Curriculum:
        from app.models.learning import Curriculum
        curriculum = Curriculum(
            session_id=session_id,
            version=version,
            curriculum_json=curriculum_json,
        )
        self.db.add(curriculum)
        await self.db.commit()
        await self.db.refresh(curriculum)
        return curriculum

    async def get_curriculum(self, session_id: UUID) -> Curriculum | None:
        from app.models.learning import Curriculum
        result = await self.db.execute(
            select(Curriculum)
            .where(Curriculum.session_id == session_id)
            .order_by(Curriculum.version.desc())
        )
        return result.scalars().first()

    async def save_topics(self, session_id: UUID, curriculum_id: UUID, weeks: list) -> list[Topic]:
        topics = []
        for week in weeks:
            for day in week.get("days", []):
                topic = Topic(
                    id=day.get("topic_id"),
                    session_id=session_id,
                    curriculum_id=curriculum_id,
                    title=day.get("title"),
                    week_number=week.get("week"),
                    day_number=day.get("day"),
                    duration_minutes=day.get("duration_minutes"),
                    status=day.get("status", "locked"),
                    search_queries=day.get("search_queries"),
                )
                self.db.add(topic)
                topics.append(topic)
        await self.db.commit()
        return topics

    async def update_topic_status(self, topic_id: str, status: str) -> Topic | None:
        topic = await self.get_topic(topic_id)
        if topic:
            topic.status = status
            if status == "completed":
                topic.completed_at = datetime.now(timezone.utc)
            await self.db.commit()
            await self.db.refresh(topic)
        return topic

    async def activate_next_topic(self, session_id: UUID, completed_topic_id: str) -> Topic | None:
        import logging
        logger = logging.getLogger(__name__)
        
        topics = await self.get_topics(session_id)
        logger.info(f"[ACTIVATE] Found {len(topics)} topics for session {session_id}")
        
        completed_topic = None
        for topic in topics:
            logger.info(f"[ACTIVATE] Topic: id={topic.id}, week={topic.week_number}, day={topic.day_number}, status={topic.status}")
            if topic.id == completed_topic_id:
                completed_topic = topic
                break
        
        if not completed_topic:
            logger.warning(f"[ACTIVATE] Completed topic {completed_topic_id} not found!")
            return None
        
        logger.info(f"[ACTIVATE] Found completed topic: {completed_topic.id} (week={completed_topic.week_number}, day={completed_topic.day_number})")
        
        next_topic = None
        for topic in topics:
            if topic.week_number == completed_topic.week_number and topic.day_number == completed_topic.day_number + 1:
                next_topic = topic
                break
        
        if not next_topic:
            logger.info(f"[ACTIVATE] No next topic in same week, looking for next week...")
            for topic in topics:
                if topic.week_number == completed_topic.week_number + 1 and topic.day_number == 1:
                    next_topic = topic
                    break
        
        if next_topic:
            logger.info(f"[ACTIVATE] Found next topic: {next_topic.id} (week={next_topic.week_number}, day={next_topic.day_number}, current_status={next_topic.status})")
            if next_topic.status in ("locked", "pending"):
                next_topic.status = "active"
                await self.db.commit()
                await self.db.refresh(next_topic)
                logger.info(f"[ACTIVATE] Activated next topic: {next_topic.id}")
            else:
                logger.info(f"[ACTIVATE] Next topic {next_topic.id} already has status: {next_topic.status}")
        else:
            logger.warning(f"[ACTIVATE] No next topic found after {completed_topic_id}")
        
        return next_topic

    async def get_topics(self, session_id: UUID) -> list[Topic]:
        result = await self.db.execute(
            select(Topic)
            .where(Topic.session_id == session_id)
            .order_by(Topic.week_number, Topic.day_number)
        )
        topics = list(result.scalars().all())

        # Self-heal: a quiz pass (score >= 0.80) is completion. Older runs
        # unlocked the next topic without flipping the current one off
        # ``active``, which left two "Aktif" cards in the curriculum UI.
        dirty = False
        for topic in topics:
            if (
                topic.status != "completed"
                and topic.quiz_score is not None
                and topic.quiz_score >= 0.80
            ):
                topic.status = "completed"
                if topic.completed_at is None:
                    topic.completed_at = datetime.now(timezone.utc)
                dirty = True
        if dirty:
            await self.db.commit()

        return topics

    async def save_module(self, topic_id: str, session_id: UUID, title: str, content_markdown: str, sources: list | None = None) -> LearningModule:
        module = LearningModule(
            topic_id=topic_id,
            session_id=session_id,
            title=title,
            content_markdown=content_markdown,
            sources=sources,
        )
        self.db.add(module)
        await self.db.commit()
        await self.db.refresh(module)
        return module

    async def get_agent_logs(self, session_id: UUID) -> list:
        from app.models.agent import AgentLog
        result = await self.db.execute(
            select(AgentLog)
            .where(AgentLog.session_id == session_id)
            .order_by(AgentLog.created_at.asc())
        )
        return list(result.scalars().all())

    async def get_resource_links(self, session_id: UUID) -> list[ResourceLink]:
        """All resource links attached to any module/topic in this session."""
        result = await self.db.execute(
            select(ResourceLink).where(ResourceLink.session_id == session_id)
        )
        return list(result.scalars().all())
