"""
Tests for the MindmapService.

Scope
-----
1. Deterministic fallback mind map builder (no LLM).
2. Mermaid syntax validation (cheapest level: structural shape check).
3. Label escaping for forbidden characters.
4. End-to-end ``get_or_generate_mindmap`` with the LLM path mocked.

The LLM is mocked because:
- Tests must run offline / in CI without a real Groq or Ollama key.
- Deterministic output is required for assertions.
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

sys.path.insert(0, '.')

from app.services.mindmap_service import (
    MindmapService,
    MindmapLLMResult,
    MINDMAP_PROMPT_TEMSynapsaTE,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_topic(topic_id, title, week, day, status="locked"):
    """Tiny Topic stand-in for testing (avoids needing a real DB row)."""
    t = MagicMock()
    t.id = topic_id
    t.title = title
    t.week_number = week
    t.day_number = day
    t.duration_minutes = 60
    t.status = status
    return t


def _make_curriculum(json_data, mindmap_json=None):
    c = MagicMock()
    c.id = "00000000-0000-0000-0000-000000000001"
    c.session_id = "00000000-0000-0000-0000-000000000002"
    c.curriculum_json = json_data
    c.mindmap_json = mindmap_json
    return c


# --------------------------------------------------------------------------- #
# Tests for the deterministic fallback
# --------------------------------------------------------------------------- #


def test_fallback_mindmap_basic_structure():
    """
    The fallback builder should always emit a syntactically valid Mermaid
    mind map with the course as the root and weeks as the second level.
    """
    topics = [
        _make_topic("t1", "Variabel & Tipe", 1, 1, "completed"),
        _make_topic("t2", "Percabangan", 1, 2, "active"),
        _make_topic("t3", "List & Dict", 2, 1, "locked"),
    ]
    result = MindmapService._build_fallback_mindmap("Python Dasar", topics)

    assert "syntax" in result
    assert "summary" in result
    syntax = result["syntax"]
    lines = syntax.splitlines()
    assert lines[0].strip() == "mindmap", f"expected first line 'mindmap', got: {lines[0]!r}"
    # Root
    assert any("Python Dasar" in l for l in lines), "course title missing from root"
    # Week 1 and Week 2 both present
    assert any("Minggu 1" in l for l in lines), "Minggu 1 missing"
    assert any("Minggu 2" in l for l in lines), "Minggu 2 missing"
    # All topics are included (by title)
    assert any("Variabel" in l for l in lines), "topic 1 missing"
    assert any("Percabangan" in l for l in lines), "topic 2 missing"
    assert any("List" in l for l in lines), "topic 3 missing"
    print("✓ test_fallback_mindmap_basic_structure passed")


def test_fallback_mindmap_escapes_parentheses():
    """
    Mermaid v11 node labels can choke on unescaped parentheses inside
    labels. The fallback must strip them so the diagram renders.
    """
    topics = [
        _make_topic("t1", "Fungsi (def) & lambda", 1, 1),
    ]
    result = MindmapService._build_fallback_mindmap("Python", topics)
    # Each line with the topic title must NOT contain raw '(' or ')'
    # (parentheses are reserved for the root icon shape)
    for line in result["syntax"].splitlines():
        if "Fungsi" in line:
            assert "(" not in line, f"parenthesis leaked: {line!r}"
            assert ")" not in line, f"parenthesis leaked: {line!r}"
    print("✓ test_fallback_mindmap_escapes_parentheses passed")


def test_fallback_mindmap_empty_topics():
    """No topics → just the root, no crash."""
    result = MindmapService._build_fallback_mindmap("Empty Course", [])
    assert result["syntax"].splitlines()[0].strip() == "mindmap"
    assert "Empty Course" in result["syntax"]
    print("✓ test_fallback_mindmap_empty_topics passed")


# --------------------------------------------------------------------------- #
# Tests for the syntax validator
# --------------------------------------------------------------------------- #


def test_validator_accepts_valid_syntax():
    syntax = "mindmap\n  root((X))\n    Minggu 1\n      Topik A"
    assert MindmapService._is_valid_mermaid_mindmap(syntax) is True
    print("✓ test_validator_accepts_valid_syntax passed")


def test_validator_rejects_garbage():
    assert MindmapService._is_valid_mermaid_mindmap("") is False
    assert MindmapService._is_valid_mermaid_mindmap("graph TD\nA-->B") is False
    assert MindmapService._is_valid_mermaid_mindmap("mindmap") is False  # <3 lines
    print("✓ test_validator_rejects_garbage passed")


# --------------------------------------------------------------------------- #
# Tests for the LLM path (LLM mocked)
# --------------------------------------------------------------------------- #


def test_get_or_generate_uses_cache_when_present():
    """
    When ``mindmap_json`` is already set on the curriculum, the service
    must return the cached value WITHOUT calling the LLM.
    """
    cached = {
        "syntax": "mindmap\n  root((Cached))\n    Minggu 1\n      T1",
        "summary": "Cached summary",
        "generated_at": "2026-06-15T00:00:00Z",
        "model": "cached",
        "node_count": 4,
    }
    curriculum = _make_curriculum(
        {"title": "Cached Course"}, mindmap_json=cached
    )
    db = MagicMock()
    svc = MindmapService(db)

    async def fake_get_session(_): return MagicMock()
    async def fake_get_curriculum(_): return curriculum
    async def fake_get_topics(_): return []  # should NOT be called

    with patch.object(MindmapService, '__init__', return_value=None):
        svc = MindmapService(db)
    svc.db = db

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(side_effect=AssertionError("must not call LLM path"))):
        result = asyncio.run(svc.get_or_generate_mindmap(
            curriculum.session_id, force_regenerate=False
        ))

    assert result is not None
    assert result["syntax"] == cached["syntax"]
    assert result["summary"] == "Cached summary"
    print("✓ test_get_or_generate_uses_cache_when_present passed")


def test_get_or_generate_falls_back_when_llm_fails():
    """
    When the LLM call raises or returns invalid syntax, the service must
    silently fall back to the deterministic mind map.
    """
    curriculum = _make_curriculum({"title": "Docker Basics"})
    topics = [
        _make_topic("t1", "Container", 1, 1),
        _make_topic("t2", "Image", 1, 2),
    ]
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()

    svc = MindmapService(db)

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)), \
         patch.object(MindmapService, '_call_llm',
                      new=AsyncMock(side_effect=RuntimeError("LLM offline"))):
        result = asyncio.run(svc.get_or_generate_mindmap(
            curriculum.session_id, force_regenerate=True
        ))

    assert result is not None
    assert result["model"] == "fallback"
    assert "Docker Basics" in result["syntax"]
    assert "Minggu 1" in result["syntax"]
    assert "Container" in result["syntax"]
    assert "Image" in result["syntax"]
    # Persisted to DB
    db.execute.assert_called_once()
    db.commit.assert_called_once()
    print("✓ test_get_or_generate_falls_back_when_llm_fails passed")


def test_get_or_generate_uses_llm_when_cache_miss():
    """
    With no cache and a working LLM, the service must use the LLM result
    (and persist it to the DB).
    """
    curriculum = _make_curriculum({"title": "ML 101", "level": "beginner"})
    topics = [
        _make_topic("t1", "Supervised Learning", 1, 1),
        _make_topic("t2", "Unsupervised Learning", 1, 2),
    ]
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    svc = MindmapService(db)

    llm_payload = {
        "syntax": "mindmap\n  root((ML 101))\n    Minggu 1\n      Supervised Learning\n      Unsupervised Learning",
        "summary": "Pengenalan machine learning.",
    }
    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)), \
         patch.object(MindmapService, '_call_llm',
                      new=AsyncMock(return_value=llm_payload)):
        result = asyncio.run(svc.get_or_generate_mindmap(
            curriculum.session_id, force_regenerate=False
        ))

    assert result is not None
    assert result["syntax"] == llm_payload["syntax"]
    assert result["summary"] == llm_payload["summary"]
    # LLM path persists to DB
    db.execute.assert_called_once()
    db.commit.assert_called_once()
    print("✓ test_get_or_generate_uses_llm_when_cache_miss passed")


def test_get_or_generate_returns_none_when_no_curriculum():
    """
    If the session has no curriculum yet, the service returns None
    (the API then returns 404).
    """
    db = MagicMock()
    svc = MindmapService(db)

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=None)):
        result = asyncio.run(svc.get_or_generate_mindmap(
            UUID("00000000-0000-0000-0000-000000000003")
        ))

    assert result is None
    print("✓ test_get_or_generate_returns_none_when_no_curriculum passed")


def test_prompt_template_substitutes_all_vars():
    """The prompt must include all placeholders correctly."""
    msg = MINDMAP_PROMPT_TEMSynapsaTE.format(
        course_title="Python",
        level="beginner",
        language="id",
        topics_json='[{"topic_id":"t1","title":"Variabel"}]',
    )
    assert "Python" in msg
    assert "beginner" in msg
    assert "topic_id" in msg
    assert "t1" in msg
    assert "Variabel" in msg
    assert "{course_title}" not in msg
    assert "{language}" not in msg
    assert "{topics_json}" not in msg
    print("✓ test_prompt_template_substitutes_all_vars passed")


# --------------------------------------------------------------------------- #
# Test runner
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    test_fallback_mindmap_basic_structure()
    test_fallback_mindmap_escapes_parentheses()
    test_fallback_mindmap_empty_topics()
    test_validator_accepts_valid_syntax()
    test_validator_rejects_garbage()
    test_get_or_generate_uses_cache_when_present()
    test_get_or_generate_falls_back_when_llm_fails()
    test_get_or_generate_uses_llm_when_cache_miss()
    test_get_or_generate_returns_none_when_no_curriculum()
    test_prompt_template_substitutes_all_vars()
    print(f"\nAll 10 MindmapService tests passed.")
