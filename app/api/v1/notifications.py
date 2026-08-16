from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.db.database import get_db
from app.services.notification_service import NotificationService
from app.schemas.notification import (
    NotificationResponse,
    NotificationPreferencesResponse,
    NotificationPreferencesUpdate,
)
from app.api.v1.auth import get_current_user
from app.models.user import User

router = APIRouter()

@router.get("", response_model=List[NotificationResponse])
async def get_notifications(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    service = NotificationService(db)
    return await service.get_user_notifications(current_user.id, limit)


@router.get("/preferences", response_model=NotificationPreferencesResponse)
async def get_notification_preferences(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Read the current user's notification preferences.

    Returns server defaults (both enabled) the first time it is read.
    """
    service = NotificationService(db)
    return await service.get_preferences(current_user.id)


@router.put("/preferences", response_model=NotificationPreferencesResponse)
async def update_notification_preferences(
    prefs_in: NotificationPreferencesUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Persist the current user's notification preferences."""
    service = NotificationService(db)
    return await service.update_preferences(
        current_user.id,
        email_enabled=prefs_in.email_enabled,
        push_enabled=prefs_in.push_enabled,
    )


@router.put("/{notification_id}/read", response_model=NotificationResponse)
async def mark_notification_as_read(
    notification_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    service = NotificationService(db)
    notification = await service.mark_as_read(notification_id, current_user.id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    return notification

@router.put("/read-all", response_model=dict)
async def mark_all_notifications_as_read(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    service = NotificationService(db)
    await service.mark_all_as_read(current_user.id)
    return {"status": "success"}

@router.delete("/{notification_id}", response_model=dict)
async def delete_notification(
    notification_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    service = NotificationService(db)
    success = await service.delete_notification(notification_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "success"}

@router.delete("", response_model=dict)
async def delete_all_notifications(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    service = NotificationService(db)
    await service.delete_all_notifications(current_user.id)
    return {"status": "success"}
