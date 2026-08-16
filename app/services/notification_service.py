from uuid import UUID
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.notification import Notification
from app.models.notification_setting import NotificationSetting

class NotificationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_preferences(self, user_id: str | UUID) -> NotificationSetting:
        """Fetch the user's notification preferences, creating a row
        with server defaults (both enabled) the first time it is read.

        A row is flushed (not committed) when missing so that the
        surrounding transaction stays intact; the caller's eventual
        commit persists it.
        """
        if isinstance(user_id, str):
            user_id = UUID(user_id)

        result = await self.db.execute(
            select(NotificationSetting).where(NotificationSetting.user_id == user_id)
        )
        prefs = result.scalar_one_or_none()
        if prefs is None:
            prefs = NotificationSetting(user_id=user_id)
            self.db.add(prefs)
            await self.db.flush()
        return prefs

    async def update_preferences(
        self,
        user_id: str | UUID,
        email_enabled: bool = True,
        push_enabled: bool = True,
    ) -> NotificationSetting:
        """Persist the user's notification preferences."""
        if isinstance(user_id, str):
            user_id = UUID(user_id)

        result = await self.db.execute(
            select(NotificationSetting).where(NotificationSetting.user_id == user_id)
        )
        prefs = result.scalar_one_or_none()
        if prefs is None:
            prefs = NotificationSetting(user_id=user_id)
            self.db.add(prefs)

        prefs.email_enabled = email_enabled
        prefs.push_enabled = push_enabled
        await self.db.commit()
        await self.db.refresh(prefs)
        return prefs

    async def get_user_notifications(self, user_id: str | UUID, limit: int = 50):
        if isinstance(user_id, str):
            user_id = UUID(user_id)
        
        result = await self.db.execute(
            select(Notification)
            .where(Notification.user_id == user_id)
            .order_by(Notification.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def create_notification(self, user_id: str | UUID, title: str, message: str, notification_type: str, link: str = None):
        if isinstance(user_id, str):
            user_id = UUID(user_id)

        # Honor the user's push preference. Missing rows default to
        # enabled, so existing users keep receiving notifications.
        prefs = await self.get_preferences(user_id)
        if prefs is not None and not prefs.push_enabled:
            return None

        notification = Notification(
            user_id=user_id,
            title=title,
            message=message,
            type=notification_type,
            link=link
        )
        self.db.add(notification)
        await self.db.commit()
        await self.db.refresh(notification)
        return notification

    async def mark_as_read(self, notification_id: str | UUID, user_id: str | UUID):
        if isinstance(notification_id, str):
            notification_id = UUID(notification_id)
        if isinstance(user_id, str):
            user_id = UUID(user_id)
            
        result = await self.db.execute(
            update(Notification)
            .where(Notification.id == notification_id, Notification.user_id == user_id)
            .values(is_read=True)
            .returning(Notification)
        )
        await self.db.commit()
        return result.scalar_one_or_none()
        
    async def mark_all_as_read(self, user_id: str | UUID):
        if isinstance(user_id, str):
            user_id = UUID(user_id)
            
        await self.db.execute(
            update(Notification)
            .where(Notification.user_id == user_id, Notification.is_read == False)
            .values(is_read=True)
        )
        await self.db.commit()

    async def delete_notification(self, notification_id: str | UUID, user_id: str | UUID):
        from sqlalchemy import delete
        if isinstance(notification_id, str):
            notification_id = UUID(notification_id)
        if isinstance(user_id, str):
            user_id = UUID(user_id)
            
        result = await self.db.execute(
            delete(Notification)
            .where(Notification.id == notification_id, Notification.user_id == user_id)
        )
        await self.db.commit()
        return result.rowcount > 0

    async def delete_all_notifications(self, user_id: str | UUID):
        from sqlalchemy import delete
        if isinstance(user_id, str):
            user_id = UUID(user_id)
            
        result = await self.db.execute(
            delete(Notification)
            .where(Notification.user_id == user_id)
        )
        await self.db.commit()
        return result.rowcount > 0
