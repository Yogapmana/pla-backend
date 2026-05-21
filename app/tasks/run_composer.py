import asyncio
import uuid
from celery import shared_task
from app.tasks.celery_app import celery_app


@celery_app.task(bind=True)
def run_composer_task(self, session_id: str, topic_id: str, raw_contents: list[dict]):
    """
    Celery task: run only the Composer Agent for a single topic.
    Useful for re-generating a specific module after feedback revision.
    """
    from app.db.database import async_sessionmaker, AsyncSession, engine
    from app.models.learning import LearningModule, ResourceLink
    from app.rag.indexer import index_module_chunks

    async def _run():
        await engine.dispose()

        from app.agents.composer import ComposerAgent
        from app.services.rag_service import get_rag_service

        composer = ComposerAgent()

        module = await composer.compose_module(
            topic_id=topic_id,
            raw_contents=raw_contents,
        )

        rag_service = get_rag_service()

        async with AsyncSession(engine) as db:
            module_rec = LearningModule(
                id=uuid.uuid4(),
                topic_id=topic_id,
                session_id=uuid.UUID(session_id),
                title=module.title,
                content_markdown=module.content_markdown,
                sources=module.sources,
            )
            db.add(module_rec)
            await db.flush()

            # Index chunks for RAG
            try:
                await index_module_chunks(
                    module=module,
                    user_id=session_id,
                    session_id=session_id,
                    db=db,
                )
            except Exception as e:
                pass

            await db.commit()
            await db.refresh(module_rec)

        await engine.dispose()
        return {
            "session_id": session_id,
            "topic_id": topic_id,
            "module_id": str(module_rec.id),
            "title": module.title,
        }

    return asyncio.run(_run())


@shared_task
def run_composer_async(session_id: str, topic_id: str, raw_contents: list[dict]):
    """Async wrapper for run_composer_task."""
    return run_composer_task.delay(session_id, topic_id, raw_contents)