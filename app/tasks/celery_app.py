from celery import Celery

from app.config import settings

celery_app = Celery(
    "pla",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.run_orchestrator",
        "app.tasks.run_planner",
        "app.tasks.run_researcher",
        "app.tasks.run_composer",
        "app.tasks.generate_module",
        "app.tasks.generate_enhanced_mindmap",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # 9 minutes soft limit
    worker_prefetch_multiplier=1,  # One task at a time per worker
    task_acks_late=True,  # Acknowledge after completion
)