"""
Test orchestrator graph compilation & conditional routing.

Verifies:
1. The graph compiles with all 4 nodes (planner, researcher, composer, replan).
2. _should_replan() routes correctly for each FeedbackAction type.
"""
import pytest
from app.agents.orchestrator import build_pla_graph, _should_replan
from app.agents.state import SynapsaState, FeedbackAction, Curriculum, LearningConfig


def test_graph_compiles_with_all_nodes():
    """Graph should compile and have 4 nodes: planner, researcher, composer, replan."""
    builder = build_pla_graph()
    graph = builder.compile()
    node_names = set(graph.nodes.keys()) if hasattr(graph, 'nodes') else set()
    # LangGraph StateGraph exposes nodes via get_graph() in newer versions
    graph_obj = graph.get_graph()
    nodes = set(graph_obj.nodes.keys())
    assert "planner" in nodes
    assert "researcher" in nodes
    assert "composer" in nodes
    assert "replan" in nodes


def test_should_replan_no_feedback():
    """Empty feedback_actions → end (no replan)."""
    state: SynapsaState = {
        "user_id": "u1",
        "session_id": "s1",
        "learning_config": None,
        "curriculum": None,
        "research_results": [],
        "modules": [],
        "chat_history": [],
        "quiz_results": [],
        "mastery_scores": {},
        "progress_signals": None,
        "feedback_actions": [],
        "agent_logs": [],
    }
    assert _should_replan(state) == "end"


def test_should_replan_continue_action():
    """'continue' action → end (user mastered topic, no revision needed)."""
    state: SynapsaState = {
        "user_id": "u1",
        "session_id": "s1",
        "learning_config": None,
        "curriculum": None,
        "research_results": [],
        "modules": [],
        "chat_history": [],
        "quiz_results": [],
        "mastery_scores": {},
        "progress_signals": None,
        "feedback_actions": [FeedbackAction(action="continue", topic_id="t1")],
        "agent_logs": [],
    }
    assert _should_replan(state) == "end"


@pytest.mark.parametrize("action", ["repeat", "review", "accelerate"])
def test_should_replan_revision_actions(action):
    """repeat/review/accelerate → replan (curriculum revision needed)."""
    state: SynapsaState = {
        "user_id": "u1",
        "session_id": "s1",
        "learning_config": None,
        "curriculum": None,
        "research_results": [],
        "modules": [],
        "chat_history": [],
        "quiz_results": [],
        "mastery_scores": {},
        "progress_signals": None,
        "feedback_actions": [FeedbackAction(action=action, topic_id="t1")],
        "agent_logs": [],
    }
    assert _should_replan(state) == "replan"


def test_should_replan_uses_latest_action():
    """When multiple feedback actions exist, only the latest one matters."""
    state: SynapsaState = {
        "user_id": "u1",
        "session_id": "s1",
        "learning_config": None,
        "curriculum": None,
        "research_results": [],
        "modules": [],
        "chat_history": [],
        "quiz_results": [],
        "mastery_scores": {},
        "progress_signals": None,
        "feedback_actions": [
            FeedbackAction(action="repeat", topic_id="t1"),
            FeedbackAction(action="continue", topic_id="t1"),  # latest
        ],
        "agent_logs": [],
    }
    assert _should_replan(state) == "end"
