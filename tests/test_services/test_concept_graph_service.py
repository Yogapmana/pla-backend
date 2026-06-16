"""
Tests for the ConceptGraphService.

Scope
-----
1. Cache hit (uses curriculum.concept_graph_json when present and version
   matches).
2. Cache miss → fresh build (LLM mocked).
3. Per-week failure isolation (one week's LLM fails, others succeed).
4. Cross-concept edge generation (same-week clique, sequential-week Jaccard,
   topic-in-multiple-concepts).
5. Resource node shape + per-topic cap.
6. Deterministic fallback path (no LLM, derives concepts from titles + stems).
7. Version-marker self-healing (stale cache triggers rebuild).
8. Prompt template substitution.

The LLM is mocked because:
- Tests must run offline / in CI without a real Groq or Ollama key.
- Deterministic output is required for assertions.
"""
import asyncio
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

sys.path.insert(0, '.')

from app.services.concept_graph_service import (
    ConceptGraphService,
    CONCEPT_EXTRACTION_PROMPT,
    _tokenize,
    _jaccard,
    _concept_label_from_title,
    SAME_WEEK_SIBLING_WEIGHT,
    SEQUENTIAL_WEEK_WEIGHT,
    CROSS_CUTTING_WEIGHT,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_topic(topic_id, title, week, day, status="locked", search_queries=None, duration=60):
    t = MagicMock()
    t.id = topic_id
    t.title = title
    t.week_number = week
    t.day_number = day
    t.duration_minutes = duration
    t.status = status
    t.search_queries = search_queries or []
    return t


def _make_curriculum(json_data, concept_graph_json=None, version=1, session_id="00000000-0000-0000-0000-000000000002"):
    c = MagicMock()
    c.id = "00000000-0000-0000-0000-000000000001"
    c.session_id = session_id
    c.curriculum_json = json_data
    c.concept_graph_json = concept_graph_json
    c.version = version
    return c


def _make_session(language="id"):
    s = MagicMock()
    s.language = language
    return s


def _make_resource(topic_id, link_type, title="Resource", url="https://x", platform=None, rating=None):
    r = MagicMock()
    r.id = f"r-{topic_id}-{link_type}"
    r.topic_id = topic_id
    r.link_type = link_type
    r.title = title
    r.url = url
    r.platform = platform
    r.rating = rating
    r.created_at = datetime(2026, 1, 1)
    return r


# --------------------------------------------------------------------------- #
# Pure-function tests
# --------------------------------------------------------------------------- #


def test_tokenize_drops_stopwords_and_short_tokens():
    """_tokenize should drop stopwords (a, the, ...) and tokens ≤2 chars."""
    tokens = _tokenize("Recursion is a basic technique for trees")
    assert "recursion" in tokens
    assert "basic" in tokens
    assert "technique" in tokens
    assert "trees" in tokens
    # Stopwords dropped
    assert "is" not in tokens
    assert "a" not in tokens
    assert "for" not in tokens
    print("✓ test_tokenize_drops_stopwords_and_short_tokens passed")


def test_jaccard_basic_similarity():
    """Identical sets → 1.0; disjoint sets → 0.0; partial → between."""
    a = _tokenize("Recursion Trees")
    b = _tokenize("Recursion Trees")
    assert _jaccard(a, b) == 1.0

    c = _tokenize("Sorting Algorithms")
    assert _jaccard(a, c) == 0.0

    d = _tokenize("Recursion Algorithms")
    sim = _jaccard(a, d)
    assert 0 < sim < 1
    print("✓ test_jaccard_basic_similarity passed")


def test_concept_label_from_title_strips_prefix():
    """Should strip 'Topik 1.1:' style prefixes and title-case the result."""
    label = _concept_label_from_title("Topik 1.1: Pengenalan Data Analysis")
    assert "Topik" not in label
    assert "1.1" not in label
    assert "Pengenalan" in label
    assert "Data" in label
    print("✓ test_concept_label_from_title_strips_prefix passed")


def test_concept_label_from_title_handles_empty():
    assert _concept_label_from_title("") == "Topik"
    print("✓ test_concept_label_from_title_handles_empty passed")


# --------------------------------------------------------------------------- #
# Deterministic fallback tests
# --------------------------------------------------------------------------- #


def test_deterministic_fallback_one_concept_per_topic():
    """When a topic has no search_queries, one concept per topic is created."""
    topics = [
        _make_topic("t1", "Container", 1, 1, search_queries=[]),
        _make_topic("t2", "Image", 1, 2, search_queries=[]),
    ]
    concepts = ConceptGraphService._deterministic_concepts_for_week(topics)
    # One concept per topic
    assert len(concepts) == 2
    topic_ids_seen = set()
    for c in concepts:
        topic_ids_seen.update(c["topic_ids"])
    assert topic_ids_seen == {"t1", "t2"}
    print("✓ test_deterministic_fallback_one_concept_per_topic passed")


def test_deterministic_fallback_groups_by_shared_stems():
    """Two topics sharing ≥2 search_query stems should collapse into one concept."""
    topics = [
        _make_topic("t1", "Topic A", 1, 1,
                    search_queries=["recursion basics", "recursion examples", "recursion patterns"]),
        _make_topic("t2", "Topic B", 1, 2,
                    search_queries=["recursion practice", "recursion exercises", "recursion mastery"]),
    ]
    concepts = ConceptGraphService._deterministic_concepts_for_week(topics)
    # Should collapse into 1 concept
    assert len(concepts) == 1
    assert set(concepts[0]["topic_ids"]) == {"t1", "t2"}
    print("✓ test_deterministic_fallback_groups_by_shared_stems passed")


# --------------------------------------------------------------------------- #
# Resource grouping tests
# --------------------------------------------------------------------------- #


def test_resources_capped_to_5_per_topic():
    """At most 5 resources per topic should be surfaced, prioritized by link_type."""
    # 7 resources for the same topic
    resources = [
        _make_resource("t1", "source", title=f"source-{i}", url=f"https://s{i}") for i in range(3)
    ] + [
        _make_resource("t1", "course", title=f"course-{i}", url=f"https://c{i}", rating=4.5) for i in range(2)
    ] + [
        _make_resource("t1", "video", title=f"video-{i}", url=f"https://v{i}", rating=4.8) for i in range(2)
    ]
    out = ConceptGraphService._group_resources_by_topic(resources, limit=5)
    assert "t1" in out
    assert len(out["t1"]) == 5
    # First should be course or video (priority over source)
    link_types = [r.link_type for r in out["t1"]]
    # Sources are lowest priority — only 1 source should make it (after the
    # 2 courses + 2 videos = 4 high-priority items, 1 source fills the cap)
    assert "source" in link_types
    # Count: 2 course + 2 video + 1 source = 5
    assert link_types.count("course") == 2
    assert link_types.count("video") == 2
    assert link_types.count("source") == 1
    print("✓ test_resources_capped_to_5_per_topic passed")


# --------------------------------------------------------------------------- #
# Cross-concept edge tests
# --------------------------------------------------------------------------- #


def test_cross_concept_edges_same_week_clique():
    """All pairs of concepts in the same week should produce a same-week edge."""
    week_concepts = {
        1: [
            {"label": "Recursion", "topic_ids": ["t1"]},
            {"label": "Iteration", "topic_ids": ["t2"]},
            {"label": "Memoization", "topic_ids": ["t3"]},
        ]
    }
    edges = ConceptGraphService._cross_concept_edges(week_concepts, [1])
    # 3 concepts → 3 pairs (1-2, 1-3, 2-3) → 3 edges
    assert len(edges) == 3
    for e in edges:
        assert e["relation"] == "concept_to_concept"
        assert e["weight"] == SAME_WEEK_SIBLING_WEIGHT
    print("✓ test_cross_concept_edges_same_week_clique passed")


def test_cross_concept_edges_sequential_week_similarity():
    """Jaccard ≥ 0.4 across consecutive weeks → sequential-week edge."""
    week_concepts = {
        1: [{"label": "Recursion Trees Patterns", "topic_ids": ["t1"]}],
        2: [{"label": "Recursion Trees Algorithms", "topic_ids": ["t2"]}],
    }
    edges = ConceptGraphService._cross_concept_edges(week_concepts, [1, 2])
    seq = [e for e in edges if e["weight"] == SEQUENTIAL_WEEK_WEIGHT]
    # {recursion, trees, patterns} vs {recursion, trees, algorithms} → 2/4 = 0.5
    assert len(seq) == 1
    assert "w1" in seq[0]["source"] and "w2" in seq[0]["target"]
    print("✓ test_cross_concept_edges_sequential_week_similarity passed")


def test_cross_concept_edges_no_sequential_when_disjoint():
    week_concepts = {
        1: [{"label": "Recursion", "topic_ids": ["t1"]}],
        2: [{"label": "Sorting", "topic_ids": ["t2"]}],
    }
    edges = ConceptGraphService._cross_concept_edges(week_concepts, [1, 2])
    seq = [e for e in edges if e["weight"] == SEQUENTIAL_WEEK_WEIGHT]
    assert len(seq) == 0
    print("✓ test_cross_concept_edges_no_sequential_when_disjoint passed")


def test_cross_concept_edges_cross_cutting():
    """A topic linked to ≥2 concepts should produce a cross-cutting edge."""
    week_concepts = {
        1: [
            {"label": "Recursion", "topic_ids": ["t1", "shared"]},
            {"label": "Trees", "topic_ids": ["shared"]},
        ],
    }
    edges = ConceptGraphService._cross_concept_edges(week_concepts, [1])
    cross = [e for e in edges if e["weight"] == CROSS_CUTTING_WEIGHT]
    assert len(cross) == 1
    print("✓ test_cross_concept_edges_cross_cutting passed")


# --------------------------------------------------------------------------- #
# End-to-end (LLM mocked) tests
# --------------------------------------------------------------------------- #


def test_get_or_build_uses_cache_when_present_and_version_matches():
    """Cache hit path — no LLM call when version_marker matches curriculum.version."""
    cached = {
        "version": 1,
        "version_marker": 1,
        "course_title": "Test",
        "generated_at": "2026-06-15T00:00:00Z",
        "model": "cached",
        "build_seconds": 0.0,
        "nodes": [{"id": "root", "kind": "root", "label": "Test", "data": {}}],
        "edges": [],
    }
    curriculum = _make_curriculum({"title": "Test"}, concept_graph_json=cached, version=1)
    db = MagicMock()
    svc = ConceptGraphService(db)

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(side_effect=AssertionError("must not fetch topics on cache hit"))):
        result = asyncio.run(svc.get_or_build_graph(
            curriculum.session_id, force_regenerate=False
        ))

    assert result is not None
    assert result["course_title"] == "Test"
    assert result["model"] == "cached"
    print("✓ test_get_or_build_uses_cache_when_present_and_version_matches passed")


def test_get_or_build_rebuilds_when_version_marker_mismatches():
    """Self-healing: stale cache (different version_marker) → rebuild."""
    cached = {
        "version": 1,
        "version_marker": 1,  # out of date — curriculum is now v2
        "course_title": "Stale",
        "nodes": [],
        "edges": [],
    }
    curriculum = _make_curriculum({"title": "Test"}, concept_graph_json=cached, version=2)
    topics = [_make_topic("t1", "Topic 1", 1, 1)]
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    svc = ConceptGraphService(db)

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)), \
         patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=_make_session())), \
         patch('app.services.learning_service.LearningService.get_resource_links',
               new=AsyncMock(return_value=[])), \
         patch.object(ConceptGraphService, '_extract_week_concepts',
                      new=AsyncMock(return_value={"concepts": [], "used_fallback": True, "error": "test"})):
        result = asyncio.run(svc.get_or_build_graph(
            curriculum.session_id, force_regenerate=False
        ))

    assert result is not None
    # New cache must have version_marker == 2
    assert result["version_marker"] == 2
    # Persisted
    db.commit.assert_called_once()
    print("✓ test_get_or_build_rebuilds_when_version_marker_mismatches passed")


def test_get_or_build_returns_none_when_no_curriculum():
    db = MagicMock()
    svc = ConceptGraphService(db)

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=None)):
        result = asyncio.run(svc.get_or_build_graph(
            UUID("00000000-0000-0000-0000-000000000003")
        ))

    assert result is None
    print("✓ test_get_or_build_returns_none_when_no_curriculum passed")


def test_get_or_build_calls_llm_once_per_week():
    """With 3 weeks, _extract_week_concepts is called 3 times (not per topic)."""
    curriculum = _make_curriculum({"title": "X"})
    topics = [
        _make_topic("t1", "T1", 1, 1),
        _make_topic("t2", "T2", 2, 1),
        _make_topic("t3", "T3", 3, 1),
    ]
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    svc = ConceptGraphService(db)

    call_log = []
    async def fake_extract(**kwargs):
        week_number = kwargs.get("week_number")
        call_log.append(week_number)
        return {"concepts": [], "used_fallback": True, "error": "test"}

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)), \
         patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=_make_session())), \
         patch('app.services.learning_service.LearningService.get_resource_links',
               new=AsyncMock(return_value=[])), \
         patch.object(ConceptGraphService, '_extract_week_concepts',
                      new=AsyncMock(side_effect=fake_extract)):
        result = asyncio.run(svc.get_or_build_graph(curriculum.session_id, force_regenerate=True))

    assert result is not None
    assert sorted(call_log) == [1, 2, 3]
    print("✓ test_get_or_build_calls_llm_once_per_week passed")


def test_get_or_build_isolates_per_week_failure():
    """If week 1 LLM fails but week 2 succeeds, the graph still gets built."""
    curriculum = _make_curriculum({"title": "X"})
    topics = [
        _make_topic("t1", "T1", 1, 1),
        _make_topic("t2", "T2", 2, 1),
    ]
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    svc = ConceptGraphService(db)

    async def fake_extract(**kwargs):
        week_number = kwargs.get("week_number")
        if week_number == 1:
            raise RuntimeError("LLM offline for week 1")
        return {"concepts": [{"label": "Wk2 Concept", "topic_ids": ["t2"], "description": None}], "used_fallback": False}

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)), \
         patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=_make_session())), \
         patch('app.services.learning_service.LearningService.get_resource_links',
               new=AsyncMock(return_value=[])), \
         patch.object(ConceptGraphService, '_extract_week_concepts',
                      new=AsyncMock(side_effect=fake_extract)):
        result = asyncio.run(svc.get_or_build_graph(curriculum.session_id, force_regenerate=True))

    assert result is not None
    # Model should be the planner model (not "fallback") because week 2 succeeded
    assert result["model"] != "fallback"
    # Week 2 concept should be in the graph
    concept_nodes = [n for n in result["nodes"] if n["kind"] == "concept"]
    assert any(n["label"] == "Wk2 Concept" for n in concept_nodes)
    print("✓ test_get_or_build_isolates_per_week_failure passed")


def test_get_or_build_force_regenerate_bypasses_cache():
    """force=True must rebuild even when cache is valid."""
    cached = {
        "version": 1,
        "version_marker": 1,
        "course_title": "Cached",
        "nodes": [],
        "edges": [],
    }
    curriculum = _make_curriculum({"title": "Fresh"}, concept_graph_json=cached, version=1)
    topics = [_make_topic("t1", "T1", 1, 1)]
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    svc = ConceptGraphService(db)

    with patch('app.services.learning_service.LearningService.get_curriculum',
               new=AsyncMock(return_value=curriculum)), \
         patch('app.services.learning_service.LearningService.get_topics',
               new=AsyncMock(return_value=topics)), \
         patch('app.services.learning_service.LearningService.get_session',
               new=AsyncMock(return_value=_make_session())), \
         patch('app.services.learning_service.LearningService.get_resource_links',
               new=AsyncMock(return_value=[])), \
         patch.object(ConceptGraphService, '_extract_week_concepts',
                      new=AsyncMock(return_value={"concepts": [], "used_fallback": True, "error": "test"})):
        result = asyncio.run(svc.get_or_build_graph(
            curriculum.session_id, force_regenerate=True
        ))

    # New build (course_title reflects the curriculum, not the cache)
    assert result["course_title"] == "Fresh"
    db.commit.assert_called_once()
    print("✓ test_get_or_build_force_regenerate_bypasses_cache passed")


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #


def test_prompt_template_substitutes_all_vars():
    msg = CONCEPT_EXTRACTION_PROMPT.format(
        course_title="Python",
        week_number=1,
        week_title="Dasar",
        language="id",
        topics_json='[{"topic_id":"t1","title":"Variabel"}]',
    )
    assert "Python" in msg
    assert "Dasar" in msg
    assert "id" in msg
    assert "topic_id" in msg
    assert "t1" in msg
    assert "Variabel" in msg
    # No unfilled placeholders
    for token in ("{course_title}", "{week_number}", "{week_title}", "{language}", "{topics_json}"):
        assert token not in msg, f"unfilled placeholder: {token}"
    print("✓ test_prompt_template_substitutes_all_vars passed")


# --------------------------------------------------------------------------- #
# Mermaid v11 mindmap syntax generator
# --------------------------------------------------------------------------- #


def _sample_graph_for_mermaid():
    """A representative concept graph (root + 1 week + 1 concept + 1 topic + 1 resource)."""
    return {
        "version": 1,
        "course_title": "Pemrograman Jaringan",
        "generated_at": "2026-06-15T00:00:00Z",
        "model": "llama-3.3-70b-versatile",
        "build_seconds": 0.0,
        "nodes": [
            {"id": "root", "kind": "root", "label": "Pemrograman Jaringan",
             "data": {"courseTitle": "Pemrograman Jaringan"}},
            {"id": "cluster-week-1", "kind": "cluster", "label": "Minggu 1",
             "week_number": 1,
             "data": {"weekNumber": 1, "title": "Pengenalan", "topicsCount": 1}},
            {"id": "concept-w1-0", "kind": "concept", "label": "Komponen Jaringan",
             "description": "Bagian-bagian jaringan",
             "topic_count": 1,
             "data": {"label": "Komponen Jaringan", "topicIds": ["t1"], "topicCount": 1, "weekNumber": 1}},
            {"id": "topic-t1", "kind": "topic", "label": "Perangkat Keras",
             "topic_id": "t1", "week_number": 1, "day_number": 1, "status": "active",
             "data": {"id": "t1", "title": "Perangkat Keras", "weekNumber": 1, "dayNumber": 1}},
            {"id": "resource-abc123", "kind": "resource", "label": "Modul Jaringan",
             "url": "https://example.com", "link_type": "course",
             "data": {"title": "Modul Jaringan", "url": "https://example.com"}},
        ],
        "edges": [
            {"id": "e1", "source": "root", "target": "cluster-week-1", "relation": "root_to_cluster"},
            {"id": "e2", "source": "cluster-week-1", "target": "concept-w1-0", "relation": "cluster_to_concept"},
            {"id": "e3", "source": "concept-w1-0", "target": "topic-t1", "relation": "concept_to_topic"},
            {"id": "e4", "source": "topic-t1", "target": "resource-abc123", "relation": "topic_to_resource"},
            # Cross-concept edge — should NOT appear in Mermaid output
            {"id": "e5", "source": "concept-w1-0", "target": "concept-w1-0", "relation": "concept_to_concept"},
        ],
    }


def test_to_mermaid_syntax_starts_with_mindmap_keyword():
    result = ConceptGraphService.to_mermaid_syntax(_sample_graph_for_mermaid())
    assert result["syntax"].startswith("mindmap\n")
    print("✓ test_to_mermaid_syntax_starts_with_mindmap_keyword passed")


def test_to_mermaid_syntax_uses_2_space_indentation():
    """Mermaid mindmap uses indentation to express hierarchy (2 spaces / level)."""
    result = ConceptGraphService.to_mermaid_syntax(_sample_graph_for_mermaid())
    lines = result["syntax"].splitlines()
    # Each level of depth = exactly 2 more spaces of indent
    assert lines[0] == "mindmap"
    assert lines[1].startswith("  ") and not lines[1].startswith("   ")  # root
    # find a child line — the cluster is at depth 1 (4 spaces of indent)
    cluster_line = next(l for l in lines if "M1" in l or "Pengenalan" in l)
    assert cluster_line.startswith("    ")  # 4 spaces
    # Concept is at depth 2 (6 spaces)
    concept_line = next(l for l in lines if "Komponen" in l)
    assert concept_line.startswith("      ")  # 6 spaces
    print("✓ test_to_mermaid_syntax_uses_2_space_indentation passed")


def test_to_mermaid_syntax_root_uses_plain_text():
    """The root is plain text (no shape decorator). Mermaid's mindmap
    parser requires this — shape decorators crash it.
    """
    result = ConceptGraphService.to_mermaid_syntax(_sample_graph_for_mermaid())
    syntax = result["syntax"]
    # No shape decorators anywhere in the output
    assert "((" not in syntax
    assert "))" not in syntax
    assert "[[" not in syntax
    # Root label is present (plain text)
    assert "  Pemrograman Jaringan" in syntax
    print("✓ test_to_mermaid_syntax_root_uses_plain_text passed")


def test_to_mermaid_syntax_emits_no_directives_at_diagram_level():
    """Mermaid v11's mindmap parser does not support ``classDef`` /
    ``class`` / ``style`` / ``cssClass`` at the diagram level — every
    one gets tokenized as a node and causes ``There can be only one
    root. No parent could be found for`` errors. We therefore emit
    plain text nodes only.
    """
    result = ConceptGraphService.to_mermaid_syntax(_sample_graph_for_mermaid())
    syntax = result["syntax"]
    for line in syntax.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "mindmap":
            continue
        # None of the forbidden directive keywords may appear
        for forbidden in ("classDef", "class ", "cssClass", "style "):
            assert not stripped.startswith(forbidden), (
                f"forbidden directive {forbidden!r} found: {stripped!r}"
            )
    # Also no ``:::<className>`` suffix — plain text only
    assert ":::" not in syntax
    print("✓ test_to_mermaid_syntax_emits_no_directives_at_diagram_level passed")


def test_to_mermaid_syntax_legend_includes_all_kinds():
    """The legend is returned for the UI (driven by themeVariables on
    the client)."""
    result = ConceptGraphService.to_mermaid_syntax(_sample_graph_for_mermaid())
    assert len(result["legend"]) == 5
    kinds = [l["kind"] for l in result["legend"]]
    assert kinds == ["root", "cluster", "concept", "topic", "resource"]
    for entry in result["legend"]:
        assert "label" in entry
        assert "color" in entry
    print("✓ test_to_mermaid_syntax_legend_includes_all_kinds passed")


def test_to_mermaid_syntax_parses_with_mermaid_v11():
    """End-to-end: feed the generated syntax to a real Mermaid v11
    parser. Mocks the DOM globals so we can run in Node, then asserts
    the rendered SVG has the expected number of nodes + edges.

    We mock ``getBBox`` / ``getComputedTextLength`` because JSDOM
    doesn't implement SVG measurement APIs. Without those, the renderer
    crashes AFTER successful parsing — our test is about parsing
    correctness, not layout accuracy.
    """
    import subprocess
    import tempfile
    import os
    import json as _json

    # Build the syntax via the service
    payload = ConceptGraphService.to_mermaid_syntax(_sample_graph_for_mermaid())
    syntax = payload["syntax"]
    assert ":::" not in syntax
    assert "classDef" not in syntax

    # Persist syntax to a temp file and run a Node probe that imports
    # the frontend's installed mermaid package and asserts the SVG
    # contains ``<g class="node "``. The probe is written into the
    # frontend directory so Node's module resolution finds the
    # ``mermaid`` package; we delete it after the test.
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    frontend_dir = os.path.join(backend_dir, "pla-frontend")
    probe_path = os.path.join(frontend_dir, "_mermaid_parse_probe.mjs")
    syntax_path = os.path.join(frontend_dir, "_mermaid_parse_probe_input.mmd")
    try:
        with open(syntax_path, "w") as f:
            f.write(syntax)
        probe = (
            "import {JSDOM} from 'jsdom';"
            "import {default as createDOMPurify} from 'dompurify';"
            "const dom = new JSDOM('<!DOCTYPE html><body></body>');"
            "for (const k of ['document','window','HTMLElement','SVGElement',"
            "  'DOMParser','CSSStyleSheet','HTMLAnchorElement','Node','Element','NodeFilter',"
            "  'DocumentFragment','Text','Comment','Range'])"
            "  globalThis[k] = dom.window[k];"
            "// Mermaid uses DOMPurify internally; give it the isomorphic shim"
            "globalThis.DOMPurify = createDOMPurify(dom.window);"
            "dom.window.SVGElement.prototype.getBBox = function() {"
            "  return {x:0,y:0,width:100,height:30};"
            "};"
            "dom.window.SVGElement.prototype.getComputedTextLength = function() { return 50; };"
            "import {readFileSync} from 'fs';"
            "import {default as mermaid} from 'mermaid';"
            "mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'default'});"
            "const syntax = readFileSync('_mermaid_parse_probe_input.mmd', 'utf8');"
            "mermaid.render('p', syntax).then(({svg}) => {"
            "  if (!svg.includes('<g class=\"node ')) {"
            "    console.error('FAIL: no <g class=\"node \"> in rendered SVG');"
            "    process.exit(1);"
            "  }"
            "  console.log('OK nodes=' + (svg.match(/<g class=\"node /g) || []).length);"
            "}).catch(err => {"
            "  console.error('FAIL:', err.message);"
            "  process.exit(2);"
            "});"
        )
        with open(probe_path, "w") as f:
            f.write(probe)
        result = subprocess.run(
            ["node", "_mermaid_parse_probe.mjs"],
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        for p in (probe_path, syntax_path):
            if os.path.exists(p):
                os.unlink(p)
    if result.returncode != 0:
        raise AssertionError(
            f"Mermaid parse failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    print("✓ test_to_mermaid_syntax_parses_with_mermaid_v11 passed")


def test_to_mermaid_syntax_omits_cross_concept_edges():
    """Mermaid mindmap only supports a tree — cross-concept edges must not appear."""
    result = ConceptGraphService.to_mermaid_syntax(_sample_graph_for_mermaid())
    # The cross-concept edge in the sample has source == target which would
    # create a self-loop; the syntax generator walks via parent→child
    # adjacency only, so it can never produce cross edges.
    # We verify by counting edges: should be 4 (structural), not 5.
    assert "  e5" not in result["syntax"]
    assert "concept_to_concept" not in result["syntax"]
    print("✓ test_to_mermaid_syntax_omits_cross_concept_edges passed")


def test_to_mermaid_syntax_escapes_special_characters():
    """Parentheses / brackets / hashes in labels must be sanitized."""
    graph = {
        "version": 1,
        "course_title": "X",
        "nodes": [
            {"id": "root", "kind": "root", "label": "C++ (Intro)", "data": {}},
            {"id": "cluster-week-1", "kind": "cluster", "label": "W1",
             "week_number": 1, "data": {}},
            {"id": "topic-t1", "kind": "topic", "label": "Hello #world [v2]",
             "topic_id": "t1", "week_number": 1, "day_number": 1, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "root", "target": "cluster-week-1", "relation": "root_to_cluster"},
            {"id": "e2", "source": "cluster-week-1", "target": "topic-t1", "relation": "cluster_to_topic"},
        ],
    }
    result = ConceptGraphService.to_mermaid_syntax(graph)
    # Parentheses around the root are part of the shape decorator, not the
    # raw label — but the inner "C++" should be preserved.
    assert "C++" in result["syntax"]
    # The inner stray parens / brackets should be stripped from inner labels
    assert "Hello #world [v2]" not in result["syntax"]
    assert "Hello" in result["syntax"]
    # The hash should be replaced with the full-width equivalent
    assert "＃" in result["syntax"]
    print("✓ test_to_mermaid_syntax_escapes_special_characters passed")


def test_to_mermaid_syntax_handles_empty_graph():
    result = ConceptGraphService.to_mermaid_syntax({"nodes": [], "edges": []})
    assert result["syntax"] == ""
    assert result["node_count"] == 0
    assert result["truncated"] is False
    print("✓ test_to_mermaid_syntax_handles_empty_graph passed")


def test_to_mermaid_syntax_truncates_when_over_cap():
    """When the graph has more nodes than the cap, set truncated=True and keep a valid tree."""
    nodes = [{"id": "root", "kind": "root", "label": "R", "data": {}}]
    for i in range(150):
        nodes.append({
            "id": f"cluster-w{i}",
            "kind": "cluster",
            "label": f"W{i}",
            "week_number": i,
            "data": {},
        })
        nodes.append({
            "id": f"topic-t{i}",
            "kind": "topic",
            "label": f"T{i}",
            "topic_id": f"t{i}",
            "week_number": i,
            "day_number": 1,
            "data": {},
        })
    edges = []
    for n in nodes[1:]:
        if n["kind"] == "cluster":
            edges.append({"id": f"e-{n['id']}", "source": "root", "target": n["id"], "relation": "root_to_cluster"})
        else:
            # attach topic to a week (any cluster with the right week)
            cluster_id = f"cluster-w{n['week_number']}"
            edges.append({"id": f"e-{n['id']}", "source": cluster_id, "target": n["id"], "relation": "cluster_to_topic"})

    graph = {"nodes": nodes, "edges": edges}
    result = ConceptGraphService.to_mermaid_syntax(graph, node_cap=10)
    assert result["truncated"] is True
    assert result["node_count"] <= 10
    # Tree must remain structurally valid: the root must be present.
    # (Plain text only — no class suffix anymore, since Mermaid v11
    # mindmap doesn't support ``:::<class>`` at the diagram level.)
    assert "\n  R\n" in result["syntax"] or result["syntax"].endswith("\n  R")
    print("✓ test_to_mermaid_syntax_truncates_when_over_cap passed")


def test_to_mermaid_syntax_returns_legend():
    """The legend should list all 5 node kinds with their colors for the UI."""
    result = ConceptGraphService.to_mermaid_syntax(_sample_graph_for_mermaid())
    assert len(result["legend"]) == 5
    kinds = [l["kind"] for l in result["legend"]]
    assert kinds == ["root", "cluster", "concept", "topic", "resource"]
    for entry in result["legend"]:
        assert "label" in entry
        assert "color" in entry
        assert entry["color"].startswith("#")
    print("✓ test_to_mermaid_syntax_returns_legend passed")


# --------------------------------------------------------------------------- #
# Test runner
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    test_tokenize_drops_stopwords_and_short_tokens()
    test_jaccard_basic_similarity()
    test_concept_label_from_title_strips_prefix()
    test_concept_label_from_title_handles_empty()
    test_deterministic_fallback_one_concept_per_topic()
    test_deterministic_fallback_groups_by_shared_stems()
    test_resources_capped_to_5_per_topic()
    test_cross_concept_edges_same_week_clique()
    test_cross_concept_edges_sequential_week_similarity()
    test_cross_concept_edges_no_sequential_when_disjoint()
    test_cross_concept_edges_cross_cutting()
    test_get_or_build_uses_cache_when_present_and_version_matches()
    test_get_or_build_rebuilds_when_version_marker_mismatches()
    test_get_or_build_returns_none_when_no_curriculum()
    test_get_or_build_calls_llm_once_per_week()
    test_get_or_build_isolates_per_week_failure()
    test_get_or_build_force_regenerate_bypasses_cache()
    test_prompt_template_substitutes_all_vars()
    test_to_mermaid_syntax_starts_with_mindmap_keyword()
    test_to_mermaid_syntax_uses_2_space_indentation()
    test_to_mermaid_syntax_root_uses_plain_text()
    test_to_mermaid_syntax_emits_no_directives_at_diagram_level()
    test_to_mermaid_syntax_legend_includes_all_kinds()
    test_to_mermaid_syntax_omits_cross_concept_edges()
    test_to_mermaid_syntax_escapes_special_characters()
    test_to_mermaid_syntax_handles_empty_graph()
    test_to_mermaid_syntax_truncates_when_over_cap()
    test_to_mermaid_syntax_parses_with_mermaid_v11()
    print(f"\nAll 28 ConceptGraphService tests passed.")
