"""
Tests for the /mindmap-data endpoint.

This endpoint returns structured curriculum data (weeks + topics) for the
interactive (non-Mermaid) mind map view. It must:

- Return 404 if no curriculum
- Return 404 if no topics
- Group topics by week_number in ascending order
- Mark the active topic correctly
- Compute completed_count and total_duration_minutes correctly
- Be deterministic (no LLM, no cache)
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

sys.path.insert(0, '.')


def _make_topic(topic_id, title, week, day, status="locked", duration=60):
    t = MagicMock()
    t.id = topic_id
    t.title = title
    t.week_number = week
    t.day_number = day
    t.duration_minutes = duration
    t.status = status
    return t


def _make_curriculum(json_data, mindmap_json=None):
    c = MagicMock()
    c.id = "00000000-0000-0000-0000-000000000001"
    c.session_id = "00000000-0000-0000-0000-000000000002"
    c.curriculum_json = json_data
    c.mindmap_json = mindmap_json
    return c


def _make_user():
    u = MagicMock()
    u.id = UUID("11111111-1111-1111-1111-111111111111")
    return u


SID = UUID("00000000-0000-0000-0000-000000000002")


def test_groups_topics_by_week_in_ascending_order():
    """
    Topics from week 3, week 1, week 2 should appear in the response
    in order week 1 → week 2 → week 3.
    """
    from app.api.v1.curriculum import get_mindmap_data

    curriculum = _make_curriculum({
        "title": "Data Analyst",
        "weeks": [
            {"week": 1, "title": "Intro"},
            {"week": 2, "title": "ML"},
            {"week": 3, "title": "Stats"},
        ],
    })
    topics = [
        _make_topic("t-3-1", "Topik 3.1", 3, 1),
        _make_topic("t-1-1", "Topik 1.1", 1, 1),
        _make_topic("t-2-1", "Topik 2.1", 2, 1),
    ]
    user = _make_user()
    db = MagicMock()

    with patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=MagicMock(user_id=user.id))), \
         patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)):
        result = asyncio.run(get_mindmap_data(
            session_id=SID, current_user=user, db=db
        ))

    assert result.total_weeks == 3
    assert result.course_title == "Data Analyst"
    assert [w.week_number for w in result.weeks] == [1, 2, 3]
    assert [w.title for w in result.weeks] == ["Intro", "ML", "Stats"]
    print("✓ test_groups_topics_by_week_in_ascending_order passed")


def test_marks_active_topic_and_completed_count():
    """
    The active_topic_id should be the first 'active' topic in the week.
    The completed_count should reflect only 'completed' status.
    """
    from app.api.v1.curriculum import get_mindmap_data

    curriculum = _make_curriculum({
        "title": "Test",
        "weeks": [{"week": 1, "title": "Minggu 1"}],
    })
    topics = [
        _make_topic("t-1-1", "Topik 1.1", 1, 1, status="completed"),
        _make_topic("t-1-2", "Topik 1.2", 1, 2, status="active"),
        _make_topic("t-1-3", "Topik 1.3", 1, 3, status="locked"),
        _make_topic("t-1-4", "Topik 1.4", 1, 4, status="locked"),
    ]
    user = _make_user()
    db = MagicMock()

    with patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=MagicMock(user_id=user.id))), \
         patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)):
        result = asyncio.run(get_mindmap_data(
            session_id=SID, current_user=user, db=db
        ))

    week1 = result.weeks[0]
    assert week1.completed_count == 1
    assert week1.active_topic_id == "t-1-2"
    assert len(week1.topics) == 4
    # Topics inside a week should be sorted by day_number
    assert [t.day_number for t in week1.topics] == [1, 2, 3, 4]
    print("✓ test_marks_active_topic_and_completed_count passed")


def test_total_duration_is_sum_of_topic_durations():
    from app.api.v1.curriculum import get_mindmap_data

    curriculum = _make_curriculum({
        "title": "X",
        "weeks": [{"week": 1, "title": "M1"}],
    })
    topics = [
        _make_topic("t1", "A", 1, 1, duration=30),
        _make_topic("t2", "B", 1, 2, duration=45),
        _make_topic("t3", "C", 1, 3, duration=60),
    ]
    user = _make_user()
    db = MagicMock()

    with patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=MagicMock(user_id=user.id))), \
         patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)):
        result = asyncio.run(get_mindmap_data(
            session_id=SID, current_user=user, db=db
        ))

    assert result.weeks[0].total_duration_minutes == 30 + 45 + 60
    print("✓ test_total_duration_is_sum_of_topic_durations passed")


def test_uses_default_week_title_when_missing():
    """
    If curriculum_json.weeks doesn't have a title for a given week_number
    (data inconsistency), fall back to 'Minggu N'.
    """
    from app.api.v1.curriculum import get_mindmap_data

    curriculum = _make_curriculum({
        "title": "X",
        "weeks": [],  # no week metadata
    })
    topics = [_make_topic("t1", "A", 1, 1)]
    user = _make_user()
    db = MagicMock()

    with patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=MagicMock(user_id=user.id))), \
         patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)):
        result = asyncio.run(get_mindmap_data(
            session_id=SID, current_user=user, db=db
        ))

    assert result.weeks[0].title == "Minggu 1"
    print("✓ test_uses_default_week_title_when_missing passed")


def test_handles_curriculum_json_with_no_weeks_array():
    """curriculum_json may be empty / partial. The endpoint must not crash."""
    from app.api.v1.curriculum import get_mindmap_data

    curriculum = _make_curriculum({})  # no weeks key
    topics = [_make_topic("t1", "A", 1, 1)]
    user = _make_user()
    db = MagicMock()

    with patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=MagicMock(user_id=user.id))), \
         patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)):
        result = asyncio.run(get_mindmap_data(
            session_id=SID, current_user=user, db=db
        ))

    assert result.total_weeks == 1
    assert result.weeks[0].title == "Minggu 1"
    print("✓ test_handles_curriculum_json_with_no_weeks_array passed")


if __name__ == "__main__":
    test_groups_topics_by_week_in_ascending_order()
    test_marks_active_topic_and_completed_count()
    test_total_duration_is_sum_of_topic_durations()
    test_uses_default_week_title_when_missing()
    test_handles_curriculum_json_with_no_weeks_array()
    print("\nAll 5 /mindmap-data tests passed.")
