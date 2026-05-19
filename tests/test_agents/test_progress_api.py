import pytest
from httpx import AsyncClient, ASGITransport
import uuid
from app.main import app
from app.db.database import get_db

class MockDB:
    async def commit(self):
        pass
    def add(self, obj):
        pass
    async def execute(self, stmt):
        class MockResult:
            def scalars(self):
                class MockScalars:
                    def all(self):
                        return []
                    def first(self):
                        # Always return None so we don't trigger replanning logic 
                        # which requires a valid curriculum_json
                        if "learning_sessions" in str(stmt):
                            from app.models.learning import LearningSession
                            return LearningSession(id=uuid.uuid4(), user_id=uuid.uuid4())
                        return None
                return MockScalars()
        return MockResult()

async def override_get_db():
    yield MockDB()

app.dependency_overrides[get_db] = override_get_db

@pytest.mark.asyncio
async def test_submit_progress_signals():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        session_id = str(uuid.uuid4())
        response = await ac.post("/api/v1/progress/signal", json={
            "session_id": session_id,
            "topic_id": "topic_test_1",
            "quiz_score": 0.8,
            "self_assessment": 0.9
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "quiz_score" in data["message"]
        assert "self_assessment" in data["message"]

@pytest.mark.asyncio
async def test_evaluate_progress_signals():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        session_id = str(uuid.uuid4())
        
        response = await ac.post("/api/v1/progress/evaluate", params={
            "session_id": session_id,
            "topic_id": "topic_test_2"
        })
        
        # When no signals in DB, mastery defaults to 0.5 -> "repeat"
        assert response.status_code == 200
        data = response.json()
        assert data["feedback_action"] == "repeat"
        assert data["mastery_score"] == 0.5
