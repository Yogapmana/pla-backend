"""
Test orchestrator graph compilation.

Curriculum replan is unwired — adaptive content uses supplementary modules.
"""
from app.agents.orchestrator import build_pla_graph


def test_graph_compiles_linear_pipeline():
    """Graph should compile with planner → researcher → composer only."""
    builder = build_pla_graph()
    graph = builder.compile()
    graph_obj = graph.get_graph()
    nodes = set(graph_obj.nodes.keys())
    assert "planner" in nodes
    assert "researcher" in nodes
    assert "composer" in nodes
    assert "replan" not in nodes
