from uuid import UUID
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.notification import Notification

class NotificationService:
    def __init__(self, db: AsyncSession):
        self.db = db

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
