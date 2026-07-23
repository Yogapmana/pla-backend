import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select
import jwt
from app.config import settings
from app.db.database import SessionLocal
from app.models.agent import AgentLog
from app.models.learning import LearningSession
from app.utils.log_broker import subscribe_to_session

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """In-process WebSocket connection registry, keyed by session_id."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id not in self.active_connections:
            return
        if websocket in self.active_connections[session_id]:
            self.active_connections[session_id].remove(websocket)
        if not self.active_connections[session_id]:
            del self.active_connections[session_id]

    async def broadcast(self, session_id: str, message: dict):
        if session_id not in self.active_connections:
            return
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


async def fetch_agent_log_history(session_uuid: UUID) -> list[dict]:
    """Load persisted agent logs for reconnect / refresh resume."""
    async with SessionLocal() as db:
        result = await db.execute(
            select(AgentLog)
            .where(AgentLog.session_id == session_uuid)
            .order_by(AgentLog.created_at.asc())
        )
        rows = list(result.scalars().all())

    return [
        {
            "type": "agent_log",
            "id": str(row.id),
            "timestamp": row.created_at.isoformat() if row.created_at else None,
            "agent": row.agent,
            "level": row.level,
            "message": row.message,
            "metadata": row.metadata_json,
        }
        for row in rows
    ]


@router.websocket("/ws/agent-log/{session_id}")
async def agent_log_stream(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time agent log streaming.

    Connect: ws://host/ws/agent-log/{session_id}?token={jwt_token}

    The server:
    1. Sends {"type": "connected", "session_id": "..."} on connect.
    2. Replays historical agent_logs from Postgres so tab-switch / refresh
       clients resume progress instead of starting from an empty stream.
    3. Subscribes to Redis pub/sub channel `pla:agent-logs:{session_id}` and
       forwards live messages for this session.
    4. Cancels the subscriber and cleans up on disconnect.
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

    # Ownership: any valid JWT must not stream another user's agent logs.
    # Check BEFORE accept so non-owners never get a live socket or history.
    try:
        async with SessionLocal() as db:
            result = await db.execute(
                select(LearningSession).where(LearningSession.id == session_uuid)
            )
            session = result.scalars().first()
    except Exception as e:
        logger.warning(f"[WS] ownership lookup failed for {session_id}: {e}")
        await websocket.close(code=1011, reason="Server error")
        return

    if session is None or str(session.user_id) != str(user_id):
        await websocket.close(code=4003, reason="Forbidden")
        return

    await manager.connect(session_id, websocket)

    async def forward_log(payload: dict) -> None:
        await websocket.send_json(payload)

    subscriber_task: asyncio.Task | None = None
    try:
        await websocket.send_json({"type": "connected", "session_id": session_id})

        # History first so the client restores stage/logs before live events.
        try:
            history = await fetch_agent_log_history(session_uuid)
            if history:
                await websocket.send_json({"type": "history", "logs": history})
        except Exception as e:
            logger.warning(f"[WS] history replay failed for {session_id}: {e}")

        subscriber_task = asyncio.create_task(
            subscribe_to_session(session_id, forward_log)
        )

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if subscriber_task is not None:
            subscriber_task.cancel()
            try:
                await subscriber_task
            except (asyncio.CancelledError, Exception):
                pass
        manager.disconnect(session_id, websocket)
