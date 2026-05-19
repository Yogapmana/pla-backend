from langgraph.graph import StateGraph, START, END
from app.agents.state import PLAState
from app.agents.planner import planner_node
from app.agents.researcher import researcher_node
from app.agents.composer import composer_node

def build_pla_graph():
    # Initialize StateGraph with our PLAState TypedDict
    builder = StateGraph(PLAState)
    
    # Add nodes for each agent
    builder.add_node("planner", planner_node)
    
    # Using a wrapper for researcher_node because it is an async function
    # LangGraph handles async nodes seamlessly
    builder.add_node("researcher", researcher_node)
    
    builder.add_node("composer", composer_node)
    
    # Define edges: START -> planner -> researcher -> composer -> END
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "composer")
    builder.add_edge("composer", END)
    
    # Compile the graph
    graph = builder.compile()
    
    return graph

# Export compiled graph for easy importing
pla_graph = build_pla_graph()
