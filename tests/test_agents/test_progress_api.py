import pytest
from httpx import AsyncClient, ASGITransport
import uuid
from app.main import app
from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User

# A single fixed identity used for every request in this test module.
# The MockDB below is wired so that any session/topic lookup resolves to
# this user, satisfying verify_session_owner / verify_topic_owner.
TEST_USER = User(id=uuid.uuid4(), email="test@example.com", username="testuser", hashed_password="x")


class MockTopicRow:
    """Stand-in for a Topic row owned by TEST_USER's session."""
    def __init__(self, session_id, topic_id):
        self.id = topic_id
        self.session_id = session_id
        self.user_id = TEST_USER.id  # so verify_topic_owner's join matches
        # Fields read by evaluate_feedback when updating a topic
        self.mastery_score = None
        self.quiz_score = None
        self.reading_time_ratio = None
        self.question_frequency_score = None
        self.self_assessment_score = None
        self.material_rating_score = None
        self.feedback_action = None


class MockSessionRow:
    """Stand-in for a LearningSession row owned by TEST_USER."""
    def __init__(self, session_id):
        self.id = session_id
        self.session_id = session_id
        self.user_id = TEST_USER.id
        self.created_at = None
        self.completed_at = None


class MockDB:
    def __init__(self):
        # callers can stash the session/topic ids they were queried with
        self.session_id = None
        self.topic_id = None

    async def commit(self):
        pass

    def add(self, obj):
        pass

    async def execute(self, stmt):
        stmt_str = str(stmt)

        class MockResult:
            def scalars(self_inner):
                class MockScalars:
                    def all(self_inner2):
                        return []

                    def first(self_inner2):
                        # Ownership guards:
                        # - verify_session_owner selects LearningSession by id
                        # - verify_topic_owner joins topics -> learning_sessions
                        #   filtered by user_id; return a row whose session_id
                        #   matches the requested session so the require_session_id
                        #   cross-check passes too.
                        if "learning_sessions" in stmt_str:
                            return MockSessionRow(self.session_id)
                        if "topics" in stmt_str:
                            return MockTopicRow(self.session_id, self.topic_id)
                        # Everything else (progress signals) -> None so
                        # evaluate_feedback runs with default mastery
                        # (0.5 -> "remedial" in current feedback_engine).
                        return None

                return MockScalars()

        return MockResult()


async def override_get_db():
    db = MockDB()
    yield db


async def override_get_current_user():
    return TEST_USER


# Override both dependencies so the secured endpoints can be exercised
# end-to-end against the mock DB without a real JWT or database.
app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user


def _patch_db_to_track_ids(session_id, topic_id):
    """Re-point the shared MockDB so its row-shaped answers agree with
    the (session_id, topic_id) the test is about to send.

    Without this, the MockDB's first() returns rows whose session_id is
    whatever the previous request set, so the topic->session scoping
    check (require_session_id) would see a mismatch.
    """
    db_factory = app.dependency_overrides[get_db]

    async def _patched():
        db = MockDB()
        db.session_id = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
        db.topic_id = topic_id
        yield db

    app.dependency_overrides[get_db] = _patched
    return db_factory


def _restore_db(db_factory):
    app.dependency_overrides[get_db] = db_factory


@pytest.mark.asyncio
async def test_submit_progress_signals():
    session_id = str(uuid.uuid4())
    saved_factory = _patch_db_to_track_ids(session_id, "topic_test_1")
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/progress/signal", json={
                "session_id": session_id,
                "topic_id": "topic_test_1",
                "quiz_score": 0.8,
                "self_assessment": 0.9
            })

            assert response.status_code == 200, response.text
            data = response.json()
            assert "quiz_score" in data["message"]
            assert "self_assessment" in data["message"]
    finally:
        _restore_db(saved_factory)


@pytest.mark.asyncio
async def test_evaluate_progress_signals():
    session_id = str(uuid.uuid4())
    saved_factory = _patch_db_to_track_ids(session_id, "topic_test_2")
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/progress/evaluate", params={
                "session_id": session_id,
                "topic_id": "topic_test_2"
            })

            # When no signals in DB, mastery defaults to 0.5 -> "remedial"
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["feedback_action"] == "remedial"
            assert data["mastery_score"] == 0.5
            assert data["message"] == "Mastery evaluated."
    finally:
        _restore_db(saved_factory)


@pytest.mark.asyncio
async def test_progress_endpoints_require_auth():
    """Regression guard: without a valid token the progress endpoints must
    reject with 401. This locks in the auth fix that previously allowed
    unauthenticated access."""
    # Drop the current_user override for this test only.
    saved_user = app.dependency_overrides.pop(get_current_user, None)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            session_id = str(uuid.uuid4())

            r1 = await ac.post("/api/v1/progress/signal", json={
                "session_id": session_id, "topic_id": "t1", "quiz_score": 0.8,
            })
            r2 = await ac.post("/api/v1/progress/evaluate", params={
                "session_id": session_id, "topic_id": "t1",
            })
            r3 = await ac.get(f"/api/v1/progress/user-metrics/{session_id}")
            r4 = await ac.get(f"/api/v1/progress/topic-unlock/{session_id}/t1")

            assert r1.status_code == 401
            assert r2.status_code == 401
            assert r3.status_code == 401
            assert r4.status_code == 401
    finally:
        if saved_user is not None:
            app.dependency_overrides[get_current_user] = saved_user
