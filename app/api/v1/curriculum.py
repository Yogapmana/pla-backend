from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.learning_service import LearningService
from app.services.mindmap_service import MindmapService
from app.services.concept_graph_service import ConceptGraphService

router = APIRouter(prefix="/curriculum", tags=["curriculum"])


class CurriculumResponse(BaseModel):
    id: str
    session_id: str
    version: int
    curriculum_json: dict
    created_at: str | None = None


class TopicResponse(BaseModel):
    id: str
    title: str
    week_number: int
    day_number: int
    duration_minutes: int
    status: str
    search_queries: list | None = None
    scheduled_date: str | None = None
    has_remedial: bool = False
    has_deep_dive: bool = False


class CurriculumDetailResponse(BaseModel):
    curriculum: CurriculumResponse | None = None
    topics: list[TopicResponse] = []


class MindmapResponse(BaseModel):
    """Response shape for the mind map endpoint."""
    session_id: str
    syntax: str
    summary: str
    generated_at: str
    model: str
    node_count: int
    cached: bool  # True if served from cache, False if freshly generated


class EnhancedMindmapResponse(BaseModel):
    """Response shape for the NotebookLM-style enhanced mindmap.

    Returned by ``GET /curriculum/{session_id}/mindmap-enhanced``.
    The ``payload`` field is the raw ``EnhancedMindmap`` dict
    produced by ``app.agents.mindmap_mapper.mindmap_v2_mapper`` —
    a 3-level structure (theme → concept → key_point) generated
    by the background Celery task that runs after the first
    module is composed. The frontend's new "Enhanced" sub-mode
    renders this directly.

    If the background task hasn't finished yet (or failed),
    ``ready`` is False and the frontend falls back to the v1
    concept graph (returned separately by /mindmap-data or
    /concept-graph).
    """
    session_id: str
    ready: bool
    payload: dict | None = None
    # If the task failed, the frontend can show a subtle hint.
    error: str | None = None


class MindmapRegenerateResponse(BaseModel):
    session_id: str
    syntax: str
    summary: str
    generated_at: str
    model: str
    node_count: int
    regenerated: bool = True


class TopicNode(BaseModel):
    """Compact topic data for the interactive mind map view."""
    id: str
    title: str
    week_number: int
    day_number: int
    status: str  # locked | active | completed
    duration_minutes: int


class WeekNode(BaseModel):
    """Week data for the interactive mind map view."""
    week_number: int
    title: str
    topics: list[TopicNode]
    total_duration_minutes: int
    completed_count: int
    active_topic_id: str | None = None


class MindmapDataResponse(BaseModel):
    """Structured curriculum data for the interactive mind map view.

    Used by the frontend to render a clean, scoped, interactive mind map
    that avoids the 70+ node clutter of a single Mermaid diagram.
    """
    session_id: str
    course_title: str
    total_weeks: int
    total_topics: int
    completed_topics: int
    weeks: list[WeekNode]


# --------------------------------------------------------------------------- #
# Concept Graph endpoint (root → cluster/week → concept → topic → resource)
# --------------------------------------------------------------------------- #


class ConceptNodeData(BaseModel):
    """A single node in the concept graph.

    Mirrors the React Flow v12 ``node.data`` shape so the frontend can
    hydrate directly.
    """
    id: str
    kind: str  # "root" | "cluster" | "concept" | "topic" | "resource"
    label: str
    # Optional fields keyed off ``kind``
    description: str | None = None
    topic_id: str | None = None
    week_number: int | None = None
    day_number: int | None = None
    status: str | None = None
    duration_minutes: int | None = None
    url: str | None = None
    link_type: str | None = None
    platform: str | None = None
    cluster_id: str | None = None
    topic_count: int | None = None
    data: dict = Field(default_factory=dict)


class ConceptEdgeData(BaseModel):
    """A single edge in the concept graph."""
    id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0


class ConceptGraphResponse(BaseModel):
    """Response shape for the structured concept graph endpoint."""
    session_id: str
    course_title: str
    generated_at: str
    model: str
    cached: bool
    build_seconds: float
    node_count: int
    edge_count: int
    nodes: list[ConceptNodeData]
    edges: list[ConceptEdgeData]


class MermaidMindmapResponse(BaseModel):
    """Response shape for the Mermaid v11 mindmap syntax endpoint.

    The ``syntax`` field is a complete Mermaid v11 mindmap source (starts
    with ``mindmap``). The frontend feeds it to ``mermaid.render()`` with
    a dark theme to get the SVG.

    ``truncated`` is True when the concept graph had more nodes than the
    Mermaid renderer's safe cap (120); the frontend should fall back to
    the Overview view in that case.
    """
    session_id: str
    course_title: str
    syntax: str
    model: str
    cached: bool
    build_seconds: float
    generated_at: str
    node_count: int
    truncated: bool
    legend: list[dict] = []


@router.get("/{session_id}", response_model=CurriculumResponse)
async def get_curriculum(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the curriculum for a learning session.
    PRD: GET /api/v1/curriculum/{session_id}
    """
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    curriculum = await service.get_curriculum(session_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found yet (still processing)")

    return CurriculumResponse(
        id=str(curriculum.id),
        session_id=str(curriculum.session_id),
        version=curriculum.version,
        curriculum_json=curriculum.curriculum_json,
        created_at=curriculum.created_at.isoformat() if curriculum.created_at else None,
    )


@router.get("/{session_id}/detail", response_model=CurriculumDetailResponse)
async def get_curriculum_detail(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get curriculum with all topics included."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")

    curriculum = await service.get_curriculum(session_id)
    topics = await service.get_topics(session_id)

    curriculum_resp = None
    if curriculum:
        curriculum_resp = CurriculumResponse(
            id=str(curriculum.id),
            session_id=str(curriculum.session_id),
            version=curriculum.version,
            curriculum_json=curriculum.curriculum_json,
            created_at=curriculum.created_at.isoformat() if curriculum.created_at else None,
        )

    from app.models.learning import LearningModule
    from sqlalchemy import select
    modules_res = await db.execute(
        select(LearningModule.topic_id, LearningModule.remedial_markdown, LearningModule.deep_dive_markdown)
        .where(LearningModule.session_id == session_id)
    )
    module_info = {row[0]: (row[1] is not None, row[2] is not None) for row in modules_res.all()}

    topic_list = [
        TopicResponse(
            id=t.id,
            title=t.title,
            week_number=t.week_number,
            day_number=t.day_number,
            duration_minutes=t.duration_minutes,
            status=t.status,
            search_queries=t.search_queries,
            scheduled_date=t.scheduled_date.isoformat() if t.scheduled_date else None,
            has_remedial=module_info.get(t.id, (False, False))[0],
            has_deep_dive=module_info.get(t.id, (False, False))[1],
        )
        for t in topics
    ]

    return CurriculumDetailResponse(
        curriculum=curriculum_resp,
        topics=topic_list,
    )


# --------------------------------------------------------------------------- #
# Mind Map endpoints
# --------------------------------------------------------------------------- #


async def _verify_session_ownership(
    session_id: UUID, current_user: User, db: AsyncSession
) -> None:
    """Raise 404 if the session doesn't exist or isn't owned by the user."""
    service = LearningService(db)
    session = await service.get_session(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Learning session not found")


async def _fetch_mindmap(
    session_id: UUID,
    current_user: User,
    db: AsyncSession,
    *,
    force_regenerate: bool,
) -> dict:
    """
    Shared helper for GET and POST mindmap endpoints.

    Returns the dict payload ready to be wrapped in a Pydantic response.
    """
    await _verify_session_ownership(session_id, current_user, db)

    mindmap_svc = MindmapService(db)
    # Check if a cached value exists (for the `cached` flag)
    learning_svc = LearningService(db)
    curriculum = await learning_svc.get_curriculum(session_id)
    had_cache = bool(curriculum is not None and curriculum.mindmap_json is not None)

    payload = await mindmap_svc.get_or_generate_mindmap(
        session_id, force_regenerate=force_regenerate
    )
    if not payload:
        raise HTTPException(
            status_code=404,
            detail="Kurikulum belum tersedia. Tunggu pipeline selesai terlebih dahulu.",
        )

    # Detect "cache hit" — payload is the same object reference we cached
    cached = had_cache and not force_regenerate
    return {
        "session_id": str(session_id),
        "syntax": payload["syntax"],
        "summary": payload.get("summary", ""),
        "generated_at": payload.get("generated_at", ""),
        "model": payload.get("model", ""),
        "node_count": payload.get("node_count", 0),
        "cached": cached,
    }


@router.get("/{session_id}/mindmap", response_model=MindmapResponse)
async def get_mindmap(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the AI-generated mind map for a learning session.

    Returns the cached mind map if available, otherwise generates a new one
    via the LLM (with a deterministic fallback if the LLM call fails).

    Response is a Mermaid v11 ``mindmap`` syntax string ready to be rendered
    by the frontend Mermaid.js renderer.
    """
    return await _fetch_mindmap(
        session_id, current_user, db, force_regenerate=False
    )


@router.post("/{session_id}/mindmap/regenerate", response_model=MindmapRegenerateResponse)
async def regenerate_mindmap(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Force-regenerate the mind map (bypasses cache).

    Useful when:
    - The user clicks a "Buat Ulang" button after a re-plan
    - The curriculum was updated and the cached map is stale
    - The user wants a fresh LLM re-grouping
    """
    result = await _fetch_mindmap(
        session_id, current_user, db, force_regenerate=True
    )
    result["regenerated"] = True
    return result


# --------------------------------------------------------------------------- #
# Mind Map DATA endpoint — used by the interactive (non-Mermaid) view
# --------------------------------------------------------------------------- #


@router.get("/{session_id}/mindmap-data", response_model=MindmapDataResponse)
async def get_mindmap_data(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return structured curriculum data for the interactive mind map view.

    Why a separate endpoint (vs. reusing /mindmap)?
    -----------------------------------------------
    The /mindmap endpoint returns an LLM-generated Mermaid ``syntax`` blob
    for a single static diagram. That single-diagram approach scales
    poorly: 70+ topics crammed into one Mermaid mind map becomes
    illegible. The interactive view (MindMapView.jsx v2) instead renders
    weeks as cards, drills into a week on click, and navigates to a
    topic — none of which require LLM output.

    So this endpoint is **cheap and deterministic** (no LLM, no cache):
    it just reads the curriculum + topics and groups them.

    Output shape::

        {
            "session_id": "...",
            "course_title": "Data Analyst",
            "total_weeks": 12,
            "total_topics": 60,
            "completed_topics": 5,
            "weeks": [
                {
                    "week_number": 1,
                    "title": "Pengenalan Data Analyst",
                    "total_duration_minutes": 300,
                    "completed_count": 5,
                    "active_topic_id": "t-...-1-3",
                    "topics": [
                        {"id": "t-...-1-1", "title": "Apa itu Data Analysis?",
                         "week_number": 1, "day_number": 1,
                         "status": "completed", "duration_minutes": 60},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    await _verify_session_ownership(session_id, current_user, db)

    learning_svc = LearningService(db)
    curriculum = await learning_svc.get_curriculum(session_id)
    if not curriculum:
        raise HTTPException(
            status_code=404,
            detail="Kurikulum belum tersedia. Tunggu pipeline selesai terlebih dahulu.",
        )

    topics = await learning_svc.get_topics(session_id)
    if not topics:
        raise HTTPException(
            status_code=404,
            detail="Topik kurikulum belum tersedia.",
        )

    cjson = curriculum.curriculum_json or {}
    course_title = cjson.get("title") or cjson.get("topic") or "Kurikulum"

    # Build a map: week_number -> title from curriculum_json.weeks
    week_titles: dict[int, str] = {}
    for w in (cjson.get("weeks") or []):
        wno = w.get("week") or w.get("week_number")
        if wno is not None:
            week_titles[int(wno)] = w.get("title") or f"Minggu {wno}"

    # Group topics by week
    by_week: dict[int, list] = {}
    for t in topics:
        # SQLAlchemy columns are typed as Column[int] at static-analysis time
        # but the runtime value is a plain int. Use a local cast to satisfy
        # Pyright without sprinkling `type: ignore` everywhere.
        wno = int(t.week_number)  # type: ignore[arg-type]
        by_week.setdefault(wno, []).append(t)

    week_nodes: list[WeekNode] = []
    total_completed = 0
    for week_no in sorted(by_week.keys()):
        week_topics = by_week[week_no]
        # Sort by day_number, then by id for stability
        week_topics_sorted = sorted(
            week_topics,
            key=lambda t: (int(t.day_number), t.id),  # type: ignore[arg-type]
        )
        completed = sum(1 for t in week_topics_sorted if t.status == "completed")
        total_completed += completed
        active_topic = next(
            (t for t in week_topics_sorted if t.status == "active"), None
        )

        week_nodes.append(
            WeekNode(
                week_number=week_no,
                title=week_titles.get(week_no) or f"Minggu {week_no}",
                total_duration_minutes=sum(
                    (t.duration_minutes or 0) for t in week_topics_sorted
                ),
                completed_count=completed,
                active_topic_id=active_topic.id if active_topic else None,
                topics=[
                    TopicNode(
                        id=t.id,
                        title=t.title,
                        week_number=t.week_number,
                        day_number=t.day_number,
                        status=t.status or "locked",
                        duration_minutes=t.duration_minutes or 0,
                    )
                    for t in week_topics_sorted
                ],
            )
        )

    return MindmapDataResponse(
        session_id=str(session_id),
        course_title=course_title,
        total_weeks=len(week_nodes),
        total_topics=len(topics),
        completed_topics=total_completed,
        weeks=week_nodes,
    )


# --------------------------------------------------------------------------- #
# Concept Graph endpoints
# --------------------------------------------------------------------------- #


@router.get(
    "/{session_id}/mindmap-enhanced",
    response_model=EnhancedMindmapResponse,
)
async def get_enhanced_mindmap(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the NotebookLM-style enhanced mindmap for a session.

    The enhanced mindmap is generated by a background Celery task
    (``app.tasks.generate_enhanced_mindmap.generate_enhanced_mindmap``)
    that runs AFTER the first module finishes composing. The task:
      1. Scrapes 1-2 real sources per topic (lightweight_researcher)
      2. Asks the LLM to organize them into a 3-level
         theme → concept → key_point structure
      3. Saves the result to ``curriculum.enhanced_mindmap_json``

    Behavior:
      - ``ready=True`` : the Celery task has finished and the
        payload is in the column. Frontend renders the
        NotebookLM-style view.
      - ``ready=False`` : the task hasn't run yet (user just
        entered the dashboard) OR failed. Frontend shows a
        "preparing mindmap..." state and polls every 5s OR
        listens for the ``mindmap_enhanced`` WebSocket event.

    The endpoint is intentionally cheap: it just reads a JSONB
    column. No LLM, no DB joins. Safe to poll.
    """
    await _verify_session_ownership(session_id, current_user, db)

    learning_svc = LearningService(db)
    curriculum = await learning_svc.get_curriculum(session_id)
    if not curriculum:
        raise HTTPException(
            status_code=404,
            detail="Kurikulum belum tersedia.",
        )

    payload = curriculum.enhanced_mindmap_json
    if not payload:
        return EnhancedMindmapResponse(
            session_id=str(session_id),
            ready=False,
            payload=None,
        )

    return EnhancedMindmapResponse(
        session_id=str(session_id),
        ready=True,
        payload=payload,
    )


@router.get("/{session_id}/concept-graph", response_model=ConceptGraphResponse)
async def get_concept_graph(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    force: bool = Query(False, description="Bypass cache and rebuild the graph."),
):
    """
    Return the structured concept graph (root → cluster → concept → topic →
    resource) for a learning session.

    The graph is cached in ``curricula.concept_graph_json``. Pass ``?force=true``
    to invalidate and rebuild. The cache also self-heals when the curriculum
    version changes (e.g. after a replan) via a ``version_marker`` field stored
    alongside the cache.
    """
    await _verify_session_ownership(session_id, current_user, db)

    learning_svc = LearningService(db)
    curriculum = await learning_svc.get_curriculum(session_id)
    if not curriculum:
        raise HTTPException(
            status_code=404,
            detail="Kurikulum belum tersedia. Tunggu pipeline selesai terlebih dahulu.",
        )

    had_cache = bool(
        isinstance(curriculum.concept_graph_json, dict)
        and curriculum.concept_graph_json.get("version") == 1
        and curriculum.concept_graph_json.get("version_marker") == curriculum.version
    )

    svc = ConceptGraphService(db)
    payload = await svc.get_or_build_graph(session_id, force_regenerate=force)
    if not payload:
        raise HTTPException(
            status_code=404,
            detail="Peta konsep sedang dibuat di latar belakang. Silakan muat ulang sebentar lagi.",
        )

    nodes = [ConceptNodeData(**n) for n in payload.get("nodes", [])]
    edges = [ConceptEdgeData(**e) for e in payload.get("edges", [])]

    return ConceptGraphResponse(
        session_id=str(session_id),
        course_title=payload.get("course_title", curriculum.curriculum_json.get("title", "Kurikulum")),
        generated_at=payload.get("generated_at", ""),
        model=payload.get("model", "fallback"),
        cached=had_cache and not force,
        build_seconds=payload.get("build_seconds", 0.0),
        node_count=len(nodes),
        edge_count=len(edges),
        nodes=nodes,
        edges=edges,
    )


@router.post("/{session_id}/concept-graph/regenerate", response_model=ConceptGraphResponse)
async def regenerate_concept_graph(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Force-regenerate the concept graph (bypasses cache)."""
    # Re-use the GET handler with force=true. We split the route so the
    # frontend can call a clean POST /regenerate pattern symmetric with
    # the legacy /mindmap/regenerate endpoint.
    return await get_concept_graph(
        session_id=session_id,
        current_user=current_user,
        db=db,
        force=True,
    )


# --------------------------------------------------------------------------- #
# Mermaid v11 mindmap endpoint
# --------------------------------------------------------------------------- #


@router.get(
    "/{session_id}/mermaid-mindmap",
    response_model=MermaidMindmapResponse,
)
async def get_mermaid_mindmap(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    force: bool = Query(False, description="Bypass the concept-graph cache."),
):
    """
    Return Mermaid v11 mindmap syntax for the session.

    The frontend renders this with ``mermaid.render()`` against a dark
    theme. The endpoint is cheap once cached — the underlying concept
    graph is shared with ``/concept-graph`` (same ``concept_graph_json``
    column, same ``version_marker`` self-healing).
    """
    await _verify_session_ownership(session_id, current_user, db)

    svc = ConceptGraphService(db)
    graph = await svc.get_or_build_graph(session_id, force_regenerate=force)
    if not graph:
        raise HTTPException(
            status_code=404,
            detail="Kurikulum belum tersedia. Tunggu pipeline selesai terlebih dahulu.",
        )

    mermaid_payload = ConceptGraphService.to_mermaid_syntax(graph)

    return MermaidMindmapResponse(
        session_id=str(session_id),
        course_title=graph.get("course_title", "Kurikulum"),
        syntax=mermaid_payload["syntax"],
        model=graph.get("model", "fallback"),
        cached=bool(graph.get("version_marker") == graph.get("version", 0)),
        build_seconds=graph.get("build_seconds", 0.0),
        generated_at=graph.get("generated_at", ""),
        node_count=mermaid_payload["node_count"],
        truncated=mermaid_payload["truncated"],
        legend=mermaid_payload["legend"],
    )


@router.post(
    "/{session_id}/mermaid-mindmap/regenerate",
    response_model=MermaidMindmapResponse,
)
async def regenerate_mermaid_mindmap(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Force-regenerate the concept graph (and therefore the Mermaid syntax)."""
    return await get_mermaid_mindmap(
        session_id=session_id,
        current_user=current_user,
        db=db,
        force=True,
    )