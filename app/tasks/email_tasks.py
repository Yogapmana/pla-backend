import asyncio
from datetime import date
from sqlalchemy.future import select
from app.tasks.celery_app import celery_app
from app.services.email_service import (
    send_welcome_email, send_daily_reminder_email, send_progress_email,
    send_verification_email, send_password_reset_email,
)
from app.db.database import SessionLocal, engine
from app.models.user import User

@celery_app.task(name="app.tasks.send_welcome_email_task")
def send_welcome_email_task(email: str, username: str):
    """Celery task to send welcome email in background"""
    send_welcome_email(email, username)
    return f"Welcome email sent to {email}"

@celery_app.task(name="app.tasks.send_verification_email_task")
def send_verification_email_task(email: str, username: str, code: str):
    """Celery task to send verification OTP email"""
    send_verification_email(email, username, code)
    return f"Verification email sent to {email}"

@celery_app.task(name="app.tasks.send_password_reset_email_task")
def send_password_reset_email_task(email: str, username: str, reset_url: str):
    """Celery task to send password reset email"""
    send_password_reset_email(email, username, reset_url)
    return f"Password reset email sent to {email}"

@celery_app.task(name="app.tasks.send_progress_email_task")
def send_progress_email_task(email: str, username: str, topic: str):
    """Celery task to send progress email in background"""
    send_progress_email(email, username, topic)
    return f"Progress email sent to {email} for {topic}"

async def _check_and_send_daily_reminders():
    engine.sync_engine.dispose(close=False)
    today = date.today()
    async with SessionLocal() as db:
        # Find users who haven't logged in today
        result = await db.execute(
            select(User).where(
                (User.last_login_date < today) | (User.last_login_date == None)
            )
        )
        users = result.scalars().all()

        for user in users:
            # We trigger the synchronous send_daily_reminder_email directly or via celery task
            # Sending directly since we are already in an async background job, but it's better to
            # dispatch a celery task for each to avoid blocking or failing halfway
            send_daily_reminder_email_task.delay(user.email, user.username)

    return len(users)

@celery_app.task(name="app.tasks.send_daily_reminder_email_task")
def send_daily_reminder_email_task(email: str, username: str):
    """Celery task to send daily reminder email in background"""
    send_daily_reminder_email(email, username)
    return f"Reminder email sent to {email}"

@celery_app.task(name="app.tasks.check_daily_reminders_task")
def check_daily_reminders_task():
    """Celery beat task to check users and send reminders"""
    # Run the async function synchronously using asyncio.run
    count = asyncio.run(_check_and_send_daily_reminders())
    return f"Checked and sent daily reminders to {count} users"
