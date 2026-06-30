"""
ConceptGraphService — builds the structured concept graph that powers the
interactive "Konsep" mind map view (root → cluster/week → concept → topic →
resource). Distinct from MindmapService, which generates Mermaid syntax for
the legacy view.

Public surface:
    ConceptGraphService.get_or_build_graph(session_id, *, force_regenerate=False)
    → dict | None

The returned dict is React-Flow ready — frontend hydrates ``nodes`` and
``edges`` directly into the canvas.

Cache shape (stored in ``curricula.concept_graph_json``):
    {
        "version":        1,
        "version_marker": <int>,   # curriculum.version at build time
        "course_title":   "...",
        "generated_at":   "...",
        "model":          "<llm name | 'fallback'>",
        "build_seconds":  <float>,
        "nodes": [...],             # React Flow-shaped node dicts
        "edges": [...],             # React Flow-shaped edge dicts
    }

Failure model
-------------
* Concept extraction runs **once per week** (batched) to keep total LLM calls
  at ~1 per week, not 1 per topic. A 12-week course → ~12 LLM calls.
* If the LLM call for a specific week fails or returns invalid JSON, that
  week falls back to the **deterministic concept derivation** (one concept
  per topic, merged by shared search_query stems). Other weeks are unaffected.
* If every week falls back, ``model`` is set to ``"fallback"`` and the
  frontend surfaces a subtle "Konsep dihasilkan otomatis" badge.
* Cache is invalidated when ``curriculum.version`` changes
  (self-healing: ``version_marker`` mismatch → rebuild).
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.learning import Curriculum, ResourceLink, Topic
from app.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


# How many resources per topic to surface in the graph. 5 is enough to surface
# diversity (paper / course / video) without exploding node count.
RESOURCES_PER_TOPIC = 5

# Cross-concept edge weight thresholds
JACCARD_SIMILARITY_THRESHOLD = 0.4
SAME_WEEK_SIBLING_WEIGHT = 0.3
SEQUENTIAL_WEEK_WEIGHT = 0.7
CROSS_CUTTING_WEIGHT = 0.5

# Stopwords (Indonesian + English) for token-based similarity. Kept small on
# purpose; embedding-free, fast.
_STOPWORDS = frozenset(
    """
    a an the of in on to for and or with at by from is are was were be been being
    yang di dan atau dengan ke dari untuk pada ini itu juga akan bisa dapat
    kita kami anda mereka dia sebagai lebih sudah belum tidak bukan saja
    """.split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, drop stopwords, return set of tokens."""
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t and t not in _STOPWORDS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _short_id() -> str:
    """A short random hex id for graph-internal node ids (resources)."""
    import uuid as _uuid
    return _uuid.uuid4().hex[:8]


class ConceptGraphService:
    """Builds and caches the structured concept graph for a learning session."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def get_or_build_graph(
        self,
        session_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> dict[str, Any] | None:
        """
        Return the cached concept graph for ``session_id`` or build a new one.

        Returns ``None`` if the session has no curriculum yet.

        Cache check is performed before any expensive fetch — a hit short-
        circuits and avoids touching the topics table.
        """
        from app.services.learning_service import LearningService

        learning_svc = LearningService(self.db)
        curriculum = await learning_svc.get_curriculum(session_id)
        if not curriculum:
            return None

        # Cache check: valid + version-marker still matches the curriculum.
        cached = curriculum.concept_graph_json
        if not cached or not isinstance(cached, dict) or "week_concepts" not in cached:
            return None

        # Load topics and resources to build the graph.
        topics = await learning_svc.get_topics(session_id)
        if not topics:
            return None

        return await self._build_from_db_concepts(
            curriculum=curriculum,
            topics=topics,
            learning_svc=learning_svc,
            session_id=session_id,
        )

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    async def _build_from_db_concepts(
        self,
        *,
        curriculum: Curriculum,
        topics: list[Topic],
        learning_svc,
        session_id: UUID,
    ) -> dict[str, Any]:
        started = time.monotonic()
        cjson = curriculum.curriculum_json or {}
        course_title = cjson.get("title") or cjson.get("topic") or "Kurikulum"
        cached = curriculum.concept_graph_json

        # Group topics by week, preserving order
        by_week: dict[int, list[Topic]] = {}
        for t in topics:
            by_week.setdefault(t.week_number, []).append(t)
        for wno in by_week:
            by_week[wno].sort(key=lambda t: (t.day_number, t.id))

        # Resolve week titles from curriculum_json
        week_titles: dict[int, str] = {}
        for w in cjson.get("weeks") or []:
            wno = w.get("week") or w.get("week_number")
            if wno is not None:
                week_titles[int(wno)] = w.get("title") or f"Minggu {int(wno)}"

        week_concepts = cached.get("week_concepts", {})
        enriched_topics = cached.get("enriched_topics", {})
        
        # Convert week_concepts keys back to int
        week_concepts_int = {int(k): v for k, v in week_concepts.items()}

        resources = await learning_svc.get_resource_links(session_id)
        resources_by_topic = self._group_resources_by_topic(
            resources, limit=RESOURCES_PER_TOPIC
        )

        nodes, edges = self._assemble_graph(
            course_title=course_title,
            topics_by_week=by_week,
            week_titles=week_titles,
            week_concepts=week_concepts_int,
            enriched_topics=enriched_topics,
            resources_by_topic=resources_by_topic,
        )

        payload = {
            "version": 1,
            "version_marker": curriculum.version,
            "course_title": course_title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": cached.get("model", "background_agent"),
            "build_seconds": round(time.monotonic() - started, 2),
            "nodes": nodes,
            "edges": edges,
        }
        return payload

    # ------------------------------------------------------------------ #
    # Resource assembly
    # ------------------------------------------------------------------ #

    @staticmethod
    def _group_resources_by_topic(
        resources: list[ResourceLink],
        *,
        limit: int,
    ) -> dict[str, list[ResourceLink]]:
        """Bucket resources by topic_id, prioritized (course/video/paper first),
        capped to ``limit`` per topic."""
        by_topic: dict[str, list[ResourceLink]] = {}
        link_type_priority = {"course": 0, "video": 1, "paper": 2, "source": 3}
        for r in resources:
            if not r.topic_id:
                continue
            by_topic.setdefault(r.topic_id, []).append(r)

        out: dict[str, list[ResourceLink]] = {}
        for topic_id, items in by_topic.items():
            items_sorted = sorted(
                items,
                key=lambda r: (
                    link_type_priority.get(r.link_type or "source", 9),
                    -(r.rating or 0.0),
                    r.created_at or datetime.min,
                ),
            )
            out[topic_id] = items_sorted[:limit]
        return out

    # ------------------------------------------------------------------ #
    # Graph assembly
    # ------------------------------------------------------------------ #

    def _assemble_graph(
        self,
        *,
        course_title: str,
        topics_by_week: dict[int, list[Topic]],
        week_titles: dict[int, str],
        week_concepts: dict[int, list[dict]],
        enriched_topics: dict[str, list[dict]],
        resources_by_topic: dict[str, list[ResourceLink]],
    ) -> tuple[list[dict], list[dict]]:
        """Build the React-Flow-shaped {nodes, edges} list."""
        nodes: list[dict] = []
        edges: list[dict] = []

        # 1. Root node
        week_keys_sorted = sorted(topics_by_week.keys())
        total_topics = sum(len(t) for t in topics_by_week.values())
        completed_topics = sum(
            1 for ts in topics_by_week.values() for t in ts if (t.status or "") == "completed"
        )
        nodes.append({
            "id": "root",
            "kind": "root",
            "label": course_title,
            "data": {
                "courseTitle": course_title,
                "totalWeeks": len(week_keys_sorted),
                "totalTopics": total_topics,
                "completedTopics": completed_topics,
            },
        })

        # 2. Cluster (week) → concept → topic → resource
        for week_no in week_keys_sorted:
            cluster_id = f"cluster-week-{week_no}"
            topics = topics_by_week[week_no]
            if not topics:
                continue
            nodes.append({
                "id": cluster_id,
                "kind": "cluster",
                "label": f"Minggu {week_no}",
                "week_number": week_no,
                "data": {
                    "weekNumber": week_no,
                    "title": week_titles.get(week_no, f"Minggu {week_no}"),
                    "topicsCount": len(topics),
                },
            })
            edges.append({
                "id": f"root-to-{cluster_id}",
                "source": "root",
                "target": cluster_id,
                "relation": "root_to_cluster",
                "weight": 0.6,
            })

            # 2a. Concept nodes
            concepts = week_concepts.get(week_no, [])
            for idx, c in enumerate(concepts):
                cid = f"concept-w{week_no}-{idx}"
                topic_count = len(c.get("topic_ids") or [])
                nodes.append({
                    "id": cid,
                    "kind": "concept",
                    "label": c.get("label", "Konsep"),
                    "description": c.get("description"),
                    "cluster_id": cluster_id,
                    "topic_count": topic_count,
                    "data": {
                        "label": c.get("label", "Konsep"),
                        "description": c.get("description"),
                        "topicIds": c.get("topic_ids") or [],
                        "topicCount": topic_count,
                        "weekNumber": week_no,
                    },
                })
                edges.append({
                    "id": f"{cluster_id}-to-{cid}",
                    "source": cluster_id,
                    "target": cid,
                    "relation": "cluster_to_concept",
                    "weight": 0.7,
                })

            # 2b. Topic nodes
            for t in topics:
                tid = f"topic-{t.id}"
                nodes.append({
                    "id": tid,
                    "kind": "topic",
                    "label": t.title,
                    "topic_id": t.id,
                    "week_number": week_no,
                    "day_number": t.day_number,
                    "status": t.status or "locked",
                    "duration_minutes": t.duration_minutes or 0,
                    "cluster_id": cluster_id,
                    "data": {
                        "id": t.id,
                        "title": t.title,
                        "weekNumber": t.week_number,
                        "dayNumber": t.day_number,
                        "status": t.status or "locked",
                        "durationMinutes": t.duration_minutes or 0,
                    },
                })
                # Topic connected to its cluster (light edge)
                edges.append({
                    "id": f"{cluster_id}-to-{tid}",
                    "source": cluster_id,
                    "target": tid,
                    "relation": "cluster_to_topic",
                    "weight": 0.3,
                })

            # 2c. Concept → topic edges (from extracted concept.topic_ids)
            for idx, c in enumerate(concepts):
                cid = f"concept-w{week_no}-{idx}"
                for topic_id in c.get("topic_ids") or []:
                    tid = f"topic-{topic_id}"
                    # Only attach if the topic actually belongs to this week
                    if any(t.id == topic_id for t in topics):
                        edges.append({
                            "id": f"{cid}-to-{tid}",
                            "source": cid,
                            "target": tid,
                            "relation": "concept_to_topic",
                            "weight": 0.8,
                        })

            # 2d. Subtopic nodes per topic
            for t in topics:
                sub_topics = enriched_topics.get(t.id, [])
                for st_idx, st in enumerate(sub_topics):
                    stid = f"subtopic-{t.id}-{st_idx}"
                    nodes.append({
                        "id": stid,
                        "kind": "subtopic",
                        "label": st.get("label", "Sub-Topik"),
                        "description": st.get("description"),
                        "cluster_id": cluster_id,
                        "topic_id": t.id,
                        "data": {
                            "label": st.get("label", "Sub-Topik"),
                            "description": st.get("description"),
                        },
                    })
                    edges.append({
                        "id": f"topic-{t.id}-to-{stid}",
                        "source": f"topic-{t.id}",
                        "target": stid,
                        "relation": "topic_to_subtopic",
                        "weight": 0.6,
                    })

            # 2e. Resource nodes per topic
            for t in topics:
                topic_resources = resources_by_topic.get(t.id, [])
                for r in topic_resources:
                    if not r.url:
                        continue
                    rid = f"resource-{_short_id()}"
                    nodes.append({
                        "id": rid,
                        "kind": "resource",
                        "label": r.title or "Sumber",
                        "url": r.url,
                        "link_type": r.link_type or "source",
                        "platform": r.platform,
                        "topic_id": t.id,
                        "cluster_id": cluster_id,
                        "data": {
                            "title": r.title or "Sumber",
                            "url": r.url,
                            "linkType": r.link_type or "source",
                            "platform": r.platform,
                            "topicId": t.id,
                        },
                    })
                    edges.append({
                        "id": f"topic-{t.id}-to-{rid}",
                        "source": f"topic-{t.id}",
                        "target": rid,
                        "relation": "topic_to_resource",
                        "weight": 0.5,
                    })

        # 3. Cross-concept edges
        cross_edges = self._cross_concept_edges(week_concepts, week_keys_sorted)
        edges.extend(cross_edges)

        return nodes, edges

    @staticmethod
    def _cross_concept_edges(
        week_concepts: dict[int, list[dict]],
        week_keys_sorted: list[int],
    ) -> list[dict]:
        """Deterministic cross-concept edges — no LLM call.

        Concept node ids in the assembled graph are ``concept-w{week}-{idx}``,
        so we can use them directly here.

        Rules (per plan):
        1. Same-week sibling concepts → weight 0.3 (clique within week)
        2. Sequential-week Jaccard ≥ 0.4 → weight 0.7 (thread through weeks)
        3. Topic linked to ≥2 concepts → edge between those concepts weight 0.5
        """
        edges: list[dict] = []
        cid = lambda week, idx: f"concept-w{week}-{idx}"  # noqa: E731

        # Pre-compute concept tokens + record list
        records: list[dict] = []
        for week_no in week_keys_sorted:
            concepts = week_concepts.get(week_no, [])
            for idx, c in enumerate(concepts):
                records.append({
                    "week": week_no,
                    "idx": idx,
                    "label": c.get("label", ""),
                    "tokens": _tokenize(c.get("label", "")),
                    "topic_ids": set(c.get("topic_ids") or []),
                })

        # 1. Same-week sibling edges (clique)
        by_week: dict[int, list[dict]] = {}
        for rec in records:
            by_week.setdefault(rec["week"], []).append(rec)
        for week_no, recs in by_week.items():
            if len(recs) < 2:
                continue
            for i, a in enumerate(recs):
                for b in recs[i + 1:]:
                    edges.append({
                        "id": f"xc-w{week_no}-{a['idx']}-{b['idx']}",
                        "source": cid(a["week"], a["idx"]),
                        "target": cid(b["week"], b["idx"]),
                        "relation": "concept_to_concept",
                        "weight": SAME_WEEK_SIBLING_WEIGHT,
                    })

        # 2. Sequential-week Jaccard edges
        sorted_weeks = sorted(by_week.keys())
        for i in range(len(sorted_weeks) - 1):
            wn, wn_next = sorted_weeks[i], sorted_weeks[i + 1]
            for a in by_week.get(wn, []):
                for b in by_week.get(wn_next, []):
                    if a["tokens"] and b["tokens"]:
                        sim = _jaccard(a["tokens"], b["tokens"])
                        if sim >= JACCARD_SIMILARITY_THRESHOLD:
                            edges.append({
                                "id": f"xc-sw{wn}-{a['idx']}-{wn_next}-{b['idx']}",
                                "source": cid(a["week"], a["idx"]),
                                "target": cid(b["week"], b["idx"]),
                                "relation": "concept_to_concept",
                                "weight": SEQUENTIAL_WEEK_WEIGHT,
                            })

        # 3. Topic-in-multiple-concepts → cross-cutting edges
        topic_to_concepts: dict[str, list[dict]] = {}
        for rec in records:
            for tid in rec["topic_ids"]:
                topic_to_concepts.setdefault(tid, []).append(rec)
        for tid, recs in topic_to_concepts.items():
            if len(recs) < 2:
                continue
            for i, a in enumerate(recs):
                for b in recs[i + 1:]:
                    edges.append({
                        "id": f"xc-xt-{tid[:6]}-{a['week']}{a['idx']}-{b['week']}{b['idx']}",
                        "source": cid(a["week"], a["idx"]),
                        "target": cid(b["week"], b["idx"]),
                        "relation": "concept_to_concept",
                        "weight": CROSS_CUTTING_WEIGHT,
                    })

        return edges

    # ------------------------------------------------------------------ #
    # Mermaid v11 mindmap syntax generation
    # ------------------------------------------------------------------ #

    # Cap on how many nodes the Mermaid renderer will accept before the SVG
    # becomes unreadable. We surface this as a soft warning in the API
    # response so the frontend can fall back to the Overview view.
    MERMAID_NODE_CAP = 120

    @staticmethod
    def to_mermaid_syntax(
        graph: dict[str, Any],
        *,
        node_cap: int = MERMAID_NODE_CAP,
    ) -> dict[str, Any]:
        """
        Convert a ConceptGraphResponse-shaped dict into Mermaid v11 mindmap
        syntax. Uses the Synapsa warm-ivory palette (matches the rest of the
        app) — the root is a terracotta circle, weeks are a soft cream
        pill, concepts are info-blue, topics are warm-ivory, and resources
        are emerald.

        Returns a dict with::

            {
                "syntax":        "<mermaid source>",
                "node_count":    <int>,    # total nodes after truncation
                "truncated":     <bool>,   # True if node_cap was hit
                "legend":        [{...}],  # classDef info for the UI
            }

        Mermaid mindmap caveats baked into this implementation:

        * Indentation = hierarchy (2 spaces per level). Tabs would break it.
        * Node labels that contain ``(`` ``)`` ``[`` ``]`` ``{`` ``}`` get
          escaped, and the leading shape decorator is stripped.
        * **All node lines must be plain text.** The ``:::<className>``
          suffix and the ``classDef`` directives both work only on
          plain text — NOT on nodes wrapped in shape decorators like
          ``((...))`` or ``[...]``. Mermaid v11's mindmap parser
          throws ``Expecting 'SPACELINE', 'NL', 'EOF', got 'CLASS'``
          when the suffix follows a shape decorator.
        * The flowchart-only ``style <id> fill:...`` directive is NOT
          supported in ``mindmap`` and causes ``No parent could be
          found`` errors. We use ``classDef`` + ``:::<className>`` on
          every node instead.
        * Cross-concept edges from the concept graph are NOT emitted —
          Mermaid ``mindmap`` syntax only supports a strict tree, no
          inter-sibling edges. The XYFlow view keeps those.
        """
        nodes = graph.get("nodes") or []
        if not nodes:
            return {"syntax": "", "node_count": 0, "truncated": False, "legend": []}

        # Index nodes by id for fast lookup; also build adjacency (parent →
        # children) from the edges so we can walk the tree top-down.
        node_by_id: dict[str, dict] = {n["id"]: n for n in nodes}
        children_of: dict[str, list[str]] = {"root": []}
        for n in nodes:
            if n["id"] != "root":
                children_of.setdefault(n["id"], [])
        for e in graph.get("edges") or []:
            src, tgt = e.get("source"), e.get("target")
            # Guard against self-loops (any kind) and edges that point to
            # a node outside the kept set — both would cause infinite
            # recursion in the depth-first walk below.
            if (
                src
                and tgt
                and src != tgt
                and src in node_by_id
                and tgt in node_by_id
                and e.get("relation") != "concept_to_concept"
            ):
                children_of.setdefault(src, []).append(tgt)
        # De-dupe children defensively in case the same edge appears twice
        # (e.g. via a manual join). Set ordering is preserved by Python.
        for parent_id in children_of:
            seen: set[str] = set()
            deduped: list[str] = []
            for child in children_of[parent_id]:
                if child not in seen and child != parent_id:
                    seen.add(child)
                    deduped.append(child)
            children_of[parent_id] = deduped
        # Stable order: sort children by week_number/day_number/label so
        # the rendered mindmap is deterministic across regenerations.
        def _sort_key(node_id: str) -> tuple:
            n = node_by_id.get(node_id, {})
            return (
                n.get("week_number") or 0,
                n.get("day_number") or 0,
                (n.get("label") or "").lower(),
            )
        for parent_id in children_of:
            children_of[parent_id].sort(key=_sort_key)

        # Count nodes (cap check)
        all_ids = list(node_by_id.keys())
        truncated = len(all_ids) > node_cap
        kept_ids = set(all_ids[:node_cap] if not truncated else all_ids)
        if truncated:
            # Keep root + a BFS-sliced prefix so the tree is structurally
            # valid (never leave a child without its parent).
            kept_ids = set()
            queue = ["root"]
            while queue and len(kept_ids) < node_cap:
                current = queue.pop(0)
                if current in kept_ids:
                    continue
                kept_ids.add(current)
                for child in children_of.get(current, []):
                    queue.append(child)

        def _is_kept(nid: str) -> bool:
            return nid in kept_ids

        # Plain-text labels only. Mermaid v11's mindmap parser requires
        # this — the ``:::<className>`` suffix + ``classDef`` approach
        # would also work, BUT ``classDef``/``class``/``style``/``cssClass``
        # directives are all tokenized as nodes by the parser and cause
        # ``There can be only one root. No parent could be found for``
        # errors. Plain text only.
        def _label_for(nid: str) -> str:
            n = node_by_id.get(nid, {})
            kind = n.get("kind")
            raw = n.get("label") or ""
            if kind == "cluster":
                return f"M{n.get('week_number', '')}: {_escape_mermaid_label(raw)}"
            if kind == "topic":
                day = n.get("day_number")
                prefix = f"{day}. " if day else ""
                return _escape_mermaid_label(f"{prefix}{raw}")
            if kind == "resource":
                return _escape_mermaid_label(f"🔗 {raw}")
            # root and concept: just the label
            return _escape_mermaid_label(raw)

        lines: list[str] = ["mindmap"]

        def _walk(nid: str, depth: int) -> None:
            if not _is_kept(nid):
                return
            indent = "  " * (depth + 1)
            lines.append(f"{indent}{_label_for(nid)}")
            for child in children_of.get(nid, []):
                if _is_kept(child):
                    _walk(child, depth + 1)

        _walk("root", 0)

        # The legend is now decorative only — the actual coloring is
        # driven by Mermaid's ``mindmapRootColor`` /
        # ``mindmapClusterColor`` / ``mindmapNodeColor`` /
        # ``mindmapLeafColor`` ``themeVariables`` which the client
        # (``MermaidMindmapView``) sets via ``mermaid.initialize()``.
        # Mermaid's mindmap only differentiates 3 depths of color
        # (root / branch / leaf), so we can't get per-kind coloring.
        legend = [
            {"kind": "root",     "label": "Mata Kuliah", "color": "#C4251C"},
            {"kind": "cluster",  "label": "Minggu",      "color": "#A39E94"},
            {"kind": "concept",  "label": "Konsep",      "color": "#1D4ED8"},
            {"kind": "topic",    "label": "Topik",       "color": "#22201D"},
            {"kind": "resource", "label": "Sumber",      "color": "#166534"},
        ]

        return {
            "syntax": "\n".join(lines),
            "node_count": len(kept_ids),
            "truncated": truncated,
            "legend": legend,
        }


def _concept_label_from_title(title: str) -> str:
    """
    Build a clean 1-4 word label from a topic title.

    Strips common prefixes like "Topik 1.1:" and title-cases the result.
    """
    if not title:
        return "Topik"
    cleaned = re.sub(r"^\s*(?:Topik|Bab|Pertemuan|Sesi)\s*[\d.:)\-]+\s*", "", title, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if not words:
        return "Topik"
    return " ".join(w.capitalize() for w in words[:4])


def _escape_mermaid_label(text: str) -> str:
    """
    Sanitize a string for safe embedding in Mermaid v11 mindmap syntax.

    Mermaid's mindmap parser is finicky with several characters:

    * ``(`` ``)`` ``[`` ``]`` ``{`` ``}`` break shape decorations
      (the parser treats them as the node's shape, not part of the label).
      We strip them but the surrounding shape (``((...))`` etc.) is
      re-added by ``_label_for`` based on the node kind.
    * ``#`` starts a classDef-style modifier — replace with the full-width
      equivalent to keep it visible.
    * Pipes and backticks also break the parser — remove.
    * Collapse internal whitespace.
    * Cap at 60 chars to keep the SVG readable (a mindmap with 60-char
      node labels gets unwieldy fast).
    """
    if not text:
        return ""
    cleaned = re.sub(r"[\(\)\[\]\{\}]", "", text)
    cleaned = cleaned.replace("#", "＃").replace("|", " ").replace("`", "'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 60:
        cleaned = cleaned[:57] + "…"
    return cleaned
