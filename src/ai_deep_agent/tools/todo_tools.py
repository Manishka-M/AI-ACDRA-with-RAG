from langchain_core.tools import tool

@tool
def write_todos(tasks: list[str]) -> list[dict]:
    """
    Converts a list of task descriptions into the internal TODO schema.
    Args:
        tasks: List of task descriptions from the Planner.
    Returns:
        List of TODO dicts: [{id, task, status}]
    """
    return [
        {"id": i, "task": task.strip(), "status": "pending"}
        for i, task in enumerate(tasks, start=1)
        if task.strip()
    ]
