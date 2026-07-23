from langgraph.graph import StateGraph, START, END
from app.agents.state import SynapsaState
from app.agents.planner import planner_node
from app.agents.researcher import researcher_node
from app.agents.composer import composer_node


def build_pla_graph():
    """
    Build the Synapsa LangGraph StateGraph.

    Flow:
        START → planner → researcher → composer → END

    Adaptive feedback does NOT replan the curriculum graph. Remedial and
    enrichment content is generated as supplementary markdown on the
    existing topic module (see ``generate_supplementary_module`` and
    quiz post-eval). ``replan_node`` remains in planner.py for reference
    but is unwired from this graph.
    """
    builder = StateGraph(SynapsaState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("composer", composer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "composer")
    builder.add_edge("composer", END)

    return builder
