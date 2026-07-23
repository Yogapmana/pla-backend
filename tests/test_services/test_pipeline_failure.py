"""Pipeline failure → session status=failed after retries exhausted."""
from unittest.mock import MagicMock

import pytest

from app.tasks.run_orchestrator import _handle_pipeline_failure


def test_handle_pipeline_failure_retries_when_attempts_remain():
    task = MagicMock()
    task.max_retries = 2
    task.request.retries = 0
    task.retry.side_effect = RuntimeError("retry-raised")

    with pytest.raises(RuntimeError, match="retry-raised"):
        _handle_pipeline_failure(task, "sess-1", ValueError("boom"))

    task.retry.assert_called_once()


def test_handle_pipeline_failure_marks_failed_on_final_attempt(monkeypatch):
    task = MagicMock()
    task.max_retries = 2
    task.request.retries = 2

    called = {}

    def fake_run(coro):
        # close the coroutine to avoid "never awaited" warnings
        coro.close()
        called["ran"] = True
        return None

    monkeypatch.setattr(
        "app.tasks.run_orchestrator.asyncio.run",
        fake_run,
    )

    result = _handle_pipeline_failure(task, "sess-final", ValueError("final boom"))

    assert result["status"] == "failed"
    assert result["session_id"] == "sess-final"
    assert "final boom" in result["error"]
    assert called.get("ran") is True
    task.retry.assert_not_called()
