"""WebSocket agent-log ownership checks."""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.websocket import agent_log_stream


class _FakeWebSocket:
    def __init__(self):
        self.closed = None
        self.accepted = False
        self.sent = []

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive_text(self):
        raise RuntimeError("should not receive when rejected")


@pytest.mark.asyncio
async def test_ws_rejects_non_owner():
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    session_id = uuid.uuid4()

    session_row = MagicMock()
    session_row.user_id = owner_id

    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = session_row
    db.execute = AsyncMock(return_value=result)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=db)
    cm.__aexit__ = AsyncMock(return_value=None)

    ws = _FakeWebSocket()
    with patch("app.api.v1.websocket.verify_ws_token", AsyncMock(return_value=str(other_id))), \
         patch("app.api.v1.websocket.SessionLocal", return_value=cm), \
         patch("app.api.v1.websocket.manager") as manager:
        await agent_log_stream(ws, str(session_id), token="tok")

    assert ws.closed == (4003, "Forbidden")
    manager.connect.assert_not_called()


@pytest.mark.asyncio
async def test_ws_rejects_missing_session():
    session_id = uuid.uuid4()
    user_id = uuid.uuid4()

    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    db.execute = AsyncMock(return_value=result)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=db)
    cm.__aexit__ = AsyncMock(return_value=None)

    ws = _FakeWebSocket()
    with patch("app.api.v1.websocket.verify_ws_token", AsyncMock(return_value=str(user_id))), \
         patch("app.api.v1.websocket.SessionLocal", return_value=cm), \
         patch("app.api.v1.websocket.manager") as manager:
        await agent_log_stream(ws, str(session_id), token="tok")

    assert ws.closed == (4003, "Forbidden")
    manager.connect.assert_not_called()


@pytest.mark.asyncio
async def test_ws_rejects_invalid_token():
    ws = _FakeWebSocket()
    with patch("app.api.v1.websocket.verify_ws_token", AsyncMock(return_value=None)), \
         patch("app.api.v1.websocket.manager") as manager:
        await agent_log_stream(ws, str(uuid.uuid4()), token="bad")

    assert ws.closed == (4001, "Invalid token")
    manager.connect.assert_not_called()
