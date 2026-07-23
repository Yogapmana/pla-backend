"""Shared persist helpers for learning pipeline Celery tasks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.db.database import AsyncSession, engine
from app.models.agent import AgentLog as DBAgentLog
from app.models.learning import (
    Curriculum,
    LearningModule,
    LearningSession,
    ResourceLink,
    Topic,
)
from app.utils.log_broker import publish_log


def resource_link_fields(res_content: Any) -> dict:
    """Map a research RawContent object to ResourceLink column kwargs."""
    link_type = "source"
    if res_content.source_type == "course":
        link_type = "course"
    elif res_content.source_type == "youtube":
        link_type = "video"
    elif res_content.source_type in ("arxiv", "semantic_scholar"):
        link_type = "paper"

    platform = None
    if res_content.source_type in ("youtube", "arxiv", "semantic_scholar"):
        platform = res_content.source_type

    price_type = rating = duration = instructor = description = relevant_section = None
    if res_content.source_type == "course" and res_content.course_metadata:
        cm = res_content.course_metadata
        platform = cm.platform
        price_type = cm.price_type
        rating = cm.rating
        duration = cm.duration
        instructor = cm.instructor
        description = cm.description
        relevant_section = cm.relevant_section

    return {
        "link_type": link_type,
        "title": res_content.source_title,
        "url": res_content.source_url,
        "platform": platform,
        "price_type": price_type,
        "rating": rating,
        "duration": duration,
        "instructor": instructor,
        "description": description,
        "relevant_section": relevant_section,
        "embed_mode": "true" if res_content.embed_mode else "false",
        "topic_id": res_content.topic_id,
    }


async def persist_new_logs(
    session_id: str,
    state_snapshot: dict,
    persisted_log_count: int,
) -> int:
    """Write only new agent logs from a graph snapshot; return updated count."""
    logs = state_snapshot.get("agent_logs") or []
    new_logs = logs[persisted_log_count:]
    if not new_logs:
        return persisted_log_count

    prepared = []
    async with AsyncSession(engine) as db:
        for log in new_logs:
            log_id = uuid.uuid4()
            db.add(
                DBAgentLog(
                    id=log_id,
                    session_id=uuid.UUID(session_id),
                    agent=log.agent,
                    level=log.level,
                    message=log.message,
                    metadata_json=log.metadata,
                )
            )
            prepared.append((log_id, log))
        await db.commit()

    for log_id, log in prepared:
        await publish_log(
            session_id,
            {
                "type": "agent_log",
                "id": str(log_id),
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "agent": log.agent,
                "level": log.level,
                "message": log.message,
                "metadata": log.metadata,
            },
        )

    return persisted_log_count + len(new_logs)


async def save_curriculum_and_topics(
    db: AsyncSession,
    session_id: str,
    curriculum: Any,
    concept_graph: dict | None,
) -> Curriculum | None:
    if not curriculum:
        return None

    if concept_graph:
        concept_graph["version_marker"] = 1

    curriculum_rec = Curriculum(
        id=uuid.uuid4(),
        session_id=uuid.UUID(session_id),
        version=1,
        curriculum_json=curriculum.model_dump(),
        concept_graph_json=concept_graph,
    )
    db.add(curriculum_rec)
    await db.flush()

    for week in curriculum.weeks:
        for day in week.days:
            db.add(
                Topic(
                    id=day.topic_id,
                    session_id=uuid.UUID(session_id),
                    curriculum_id=curriculum_rec.id,
                    title=day.title,
                    week_number=week.week,
                    day_number=day.day,
                    duration_minutes=day.duration_minutes,
                    status=day.status,
                    search_queries=day.search_queries,
                )
            )
    await db.flush()
    return curriculum_rec


async def save_modules_and_links(
    db: AsyncSession,
    session_id: str,
    modules: list,
    research_results: list,
) -> dict:
    """Persist modules + resource links. Returns topic_id → module_id map."""
    topic_to_module_id = {}
    for module in modules:
        mod_id = uuid.uuid4()
        topic_to_module_id[module.topic_id] = mod_id
        db.add(
            LearningModule(
                id=mod_id,
                topic_id=module.topic_id,
                session_id=uuid.UUID(session_id),
                title=module.title,
                content_markdown=module.content_markdown,
                sources=module.sources,
            )
        )
    await db.flush()

    for res_content in research_results:
        fields = resource_link_fields(res_content)
        topic_id = fields.pop("topic_id")
        db.add(
            ResourceLink(
                id=uuid.uuid4(),
                module_id=topic_to_module_id.get(topic_id),
                topic_id=topic_id,
                session_id=uuid.UUID(session_id),
                **fields,
            )
        )
    await db.flush()
    return topic_to_module_id


async def mark_session_status(
    db: AsyncSession,
    session_id: str,
    *,
    ready: bool,
) -> None:
    result = await db.execute(
        select(LearningSession).where(LearningSession.id == uuid.UUID(session_id))
    )
    session = result.scalars().first()
    if not session:
        return
    if ready:
        session.status = "ready"
        session.completed_at = datetime.utcnow()
    else:
        session.status = "waiting_approval"
