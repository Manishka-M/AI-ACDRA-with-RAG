from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages:          Annotated[list, add_messages]
    user_query:        str
    todos:             list[dict]
    planning_complete: bool
    completed_tasks:   list[dict]
    final_answer:      str
    writer_used:       bool
    retry_counts:      dict[int, int]
    execution_log:     list[dict[str, Any]]
    sources:           list[dict[str, str]]
