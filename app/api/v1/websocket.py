import asyncio
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import jwt
from app.config import settings
from app.db.database import engine
from app.models.agent import AgentLog

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            dead = []
            for ws in self.active_connections[session_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(session_id, ws)


manager = ConnectionManager()


async def verify_ws_token(token: str) -> str | None:
    """Verify JWT token and return user_id."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"],
        )
        user_id: str = payload.get("sub")
        return user_id
    except jwt.PyJWTError:
        return None


async def poll_logs(session_id: UUID, user_id: UUID, websocket: WebSocket, known_log_count: int):
    """
    Poll agent_logs table for new logs since last check.
    Runs as a background task per WebSocket connection.
    """
    async with AsyncSession(engine) as db:
        result = await db.execute(
            select(AgentLog)
            .where(AgentLog.session_id == session_id)
            .order_by(AgentLog.created_at.asc())
        )
        all_logs = list(result.scalars().all())

    new_logs = all_logs[known_log_count:]
    for log in new_logs:
        try:
            await websocket.send_json({
                "timestamp": log.created_at.isoformat() if log.created_at else "",
                "agent": log.agent,
                "level": log.level,
                "message": log.message,
                "metadata": log.metadata_json or {},
            })
        except WebSocketDisconnect:
            break
        known_log_count += 1

    return known_log_count


@router.websocket("/ws/agent-log/{session_id}")
async def agent_log_stream(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time agent log streaming.

    Connect: ws://host/ws/agent-log/{session_id}?token={jwt_token}
    The server sends JSON objects:
        {"timestamp": "...", "agent": "planner", "level": "info", "message": "...", "metadata": {}}
    """
    user_id = await verify_ws_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token")
        return

    try:
        session_uuid = UUID(session_id)
    except ValueError:
        await websocket.close(code=4002, reason="Invalid session_id")
        return

    await manager.connect(session_id, websocket)
    known_count = 0

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
        })

        # Poll for new logs every 1 second
        while True:
            try:
                known_count = await poll_logs(session_uuid, UUID(user_id), websocket, known_count)
                await asyncio.sleep(1.0)
            except WebSocketDisconnect:
                break
    except Exception:
        pass
    finally:
        manager.disconnect(session_id, websocket)