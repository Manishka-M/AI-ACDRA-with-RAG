from ai_deep_agent.agents.supervisor import run_supervisor
from ai_deep_agent.state.state import AgentState

def supervisor_node(state: AgentState) -> AgentState:
    return run_supervisor(state)

def should_continue(state: AgentState) -> str:
    return "end"
