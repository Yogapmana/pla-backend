import asyncio
import uuid
from celery import shared_task
from app.tasks.celery_app import celery_app


@celery_app.task(bind=True)
def run_researcher_task(self, session_id: str, search_queries: list[dict]):
    """
    Celery task: run only the Researcher Agent to gather raw content.
    search_queries: list of dicts with keys {topic_id, queries: list[str]}
    Returns list of RawContent objects for storage.
    """
    from app.db.database import async_sessionmaker, AsyncSession, engine

    async def _run():
        await engine.dispose()

        from app.agents.researcher import ResearcherAgent
        from app.models.learning import ResourceLink

        researcher = ResearcherAgent()
        all_results = []

        async with AsyncSession(engine) as db:
            for topic_query in search_queries:
                topic_id = topic_query["topic_id"]
                queries = topic_query["queries"]

                raw_contents = await researcher.search_all(
                    topic_id=topic_id,
                    queries=queries,
                )

                for res_content in raw_contents:
                    link_type = "source"
                    if res_content.source_type == "course":
                        link_type = "course"
                    elif res_content.source_type == "youtube":
                        link_type = "video"
                    elif res_content.source_type in ("arxiv", "semantic_scholar"):
                        link_type = "paper"

                    platform = None
                    if res_content.source_type == "youtube":
                        platform = "youtube"
                    elif res_content.source_type == "arxiv":
                        platform = "arxiv"
                    elif res_content.source_type == "semantic_scholar":
                        platform = "semantic_scholar"

                    price_type = None
                    rating = None
                    duration = None
                    instructor = None
                    description = None
                    relevant_section = None

                    if res_content.source_type == "course" and res_content.course_metadata:
                        cm = res_content.course_metadata
                        platform = cm.platform
                        price_type = cm.price_type
                        rating = cm.rating
                        duration = cm.duration
                        instructor = cm.instructor
                        description = cm.description
                        relevant_section = cm.relevant_section

                    link_rec = ResourceLink(
                        id=uuid.uuid4(),
                        module_id=None,
                        topic_id=res_content.topic_id,
                        session_id=uuid.UUID(session_id),
                        link_type=link_type,
                        title=res_content.source_title,
                        url=res_content.source_url,
                        platform=platform,
                        price_type=price_type,
                        rating=rating,
                        duration=duration,
                        instructor=instructor,
                        description=description,
                        relevant_section=relevant_section,
                        embed_mode=res_content.embed_mode,
                    )
                    db.add(link_rec)
                    all_results.append(res_content)

                await db.flush()

            await db.commit()

        await engine.dispose()
        return {
            "session_id": session_id,
            "results_count": len(all_results),
        }

    return asyncio.run(_run())


@shared_task
def run_researcher_async(session_id: str, search_queries: list[dict]):
    """Async wrapper for run_researcher_task."""
    return run_researcher_task.delay(session_id, search_queries)