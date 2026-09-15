from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from ai_deep_agent.llms.factory import get_llm
from ai_deep_agent.memory.virtual_fs import workspace

def _x(resp) -> str:
    c = resp.content
    if isinstance(c, list):
        return " ".join(p.get("text","") if isinstance(p,dict) else str(p) for p in c).strip()
    return str(c).strip()

_CODING_PROMPT = """You are the Coding Agent of an autonomous AI system.
You write production-quality code with full explanations.
Prefer Java for placement/interview problems unless specified otherwise.

STRUCTURE:

## Problem Understanding
Restate what you are implementing and the key constraints.

## Approach
Explain your chosen algorithm/data structure and WHY it is optimal.
Briefly mention alternatives you considered and why you rejected them.

## Implementation
Provide the complete, runnable code.
- Proper class structure, method signatures, and imports
- Meaningful variable names
- Inline comments on non-obvious logic

```java
// Complete implementation here
Complexity Analysis
Time complexity: O(...) with justification
Space complexity: O(...) with justification
Edge Cases
List and explain how each edge case is handled.
Test Cases
Provide 3-5 test cases with expected output.
RULES:
NEVER write pseudocode when actual implementation is expected.
NEVER claim code was tested unless it was actually executed.
If context is insufficient, state what assumption you made."""



_llm = get_llm(role="coding")
def run_coding(
    task: str,
    task_id: int,
    feedback: str = "",
    previous_output: str = "",
    rag_context: str = "",           # ← 1. NEW parameter
    ) -> dict[str, Any]:
    ctx = workspace.build_context()
    # ← 2. RAG section
    rag_section = (
    f"\n\n Relevant Documents (use for problem specs/constraints):\n{rag_context[:1500]}\n"
    if rag_context.strip() else ""
    )
    messages = [
    SystemMessage(content=_CODING_PROMPT),
    HumanMessage(content=(
    f"Task:\n{task}\n\n"
    + (f"Previous attempt (improve on this):\n{previous_output}\n\n" if previous_output else "")
    + (f"Specific corrections needed:\n{feedback}\n\n"               if feedback        else "")
    + rag_section                                                      # ← 3. inject
    + (f"Available context:\n{ctx}" if ctx != "(workspace empty)" else "")
    )),
    ]
    result   = _x(_llm.invoke(messages))
    filename = f"coding_task{task_id}.md"
    workspace.write(filename, result)
    return {"result": result, "filename": filename}