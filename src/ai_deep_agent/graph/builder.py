from langgraph.graph import StateGraph, END
from ai_deep_agent.state.state import AgentState
from ai_deep_agent.graph.nodes import supervisor_node, should_continue

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", should_continue, {"end": END})
    return g.compile()

app = build_graph()
