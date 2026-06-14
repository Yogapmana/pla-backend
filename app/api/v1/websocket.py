import asyncio
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import jwt
from app.config import settings
from app.utils.log_broker import subscribe_to_session

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
        if session_id in self.active_connections:
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
    2. Subscribes to Redis pub/sub channel `pla:agent-logs:{session_id}` and
       forwards every message to the in-process ConnectionManager, which
       broadcasts to all WebSockets for that session.
    3. Runs the subscriber as a background asyncio task so the main coroutine
       can keep the connection alive.
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

    await manager.connect(session_id, websocket)

    # Forwarder callback: receives a parsed log dict from the broker and
    # pushes it to every WebSocket subscribed to this session_id in this
    # process. This is safe to call concurrently from the broker task.
    async def forward_log(payload: dict) -> None:
        await manager.broadcast(session_id, payload)

    subscriber_task: asyncio.Task | None = None
    try:
        await websocket.send_json({"type": "connected", "session_id": session_id})

        # Spawn the Redis subscriber as a background task.
        subscriber_task = asyncio.create_task(
            subscribe_to_session(session_id, forward_log)
        )

        # Keep the connection alive. The subscriber runs concurrently and
        # pushes messages via the WebSocket. We just need to wait for the
        # client to disconnect (or for an error to occur).
        while True:
            # The client isn't expected to send anything meaningful here,
            # but reading lets us detect a clean close vs a network drop.
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
