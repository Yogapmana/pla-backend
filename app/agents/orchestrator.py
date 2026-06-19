from langgraph.graph import StateGraph, START, END
from app.agents.state import PLAState
from app.agents.planner import planner_node, replan_node
from app.agents.researcher import researcher_node
from app.agents.composer import composer_node


def _should_replan(state: PLAState) -> str:
    """
    Conditional edge after composer: invoke replan only when feedback has
    been recorded AND the action requires curriculum revision.

    - "continue"  → skip replan (user has mastered the topic, proceed as planned)
    - "repeat"    → replan with simpler material
    - "review"    → replan to add a review session
    - "accelerate"→ replan to skip ahead
    - empty list  → skip replan (initial pipeline, no feedback yet)
    """
    feedback_actions = state.get("feedback_actions") or []
    if not feedback_actions:
        return "end"

    latest = feedback_actions[-1]
    action = getattr(latest, "action", None)
    if action in ("repeat", "review", "accelerate"):
        return "replan"
    # "continue" or any unknown action — proceed to end without revising
    return "end"


def build_pla_graph():
    """
    Build the PLA LangGraph StateGraph.

    Flow:
        START → planner → researcher → composer → (should_replan?)
                                                       ├─ replan → END
                                                       └─ END

    - Initial run: feedback_actions is empty → composer → END
    - Resume run after evaluate endpoint populated feedback_actions:
        composer → replan (only for repeat/review/accelerate) → END

    Note on the mindmap
    -------------------
    Earlier versions of this graph had a parallel ``mindmap_mapper``
    branch that ran right after the planner. That node only had the
    topic TITLES to work with (the researcher hadn't fetched
    anything yet), so the output was shallow — a flat list of
    concepts with 1-4-word labels. The mindmap was visible early
    but low quality.

    The current pipeline does NOT generate the mindmap here. Instead,
    a separate background Celery task
    (``app.tasks.generate_enhanced_mindmap``) runs LATER — right
    after the first module is composed — once the lightweight
    researcher has scraped 1-2 real sources per topic. The output
    is a 3-level NotebookLM-style structure (theme → concept →
    key_point) saved to ``curriculum.enhanced_mindmap_json``.

    The user experience: they can enter the dashboard as soon as
    module 1 is ready, and the enhanced mindmap arrives a few
    seconds later while they're already exploring. By the time
    they look at the Curriculum page, the new mindmap is usually
    visible.
    """
    builder = StateGraph(PLAState)

    # Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("composer", composer_node)
    builder.add_node("replan", replan_node)

    # Linear edges for the generation pipeline
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "composer")

    # Conditional: replan only if feedback demands curriculum revision
    builder.add_conditional_edges(
        "composer",
        _should_replan,
        {
            "replan": "replan",
            "end": END,
        },
    )
    builder.add_edge("replan", END)

    return builder
