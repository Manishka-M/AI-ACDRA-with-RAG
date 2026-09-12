from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from ai_deep_agent.llms.factory import get_llm
from ai_deep_agent.memory.virtual_fs import workspace

def _x(resp) -> str:
    """Extract text from LLM response (handles Gemini list format)."""
    c = resp.content
    if isinstance(c, list):
        return " ".join(p.get("text","") if isinstance(p,dict) else str(p) for p in c).strip()
    return str(c).strip()



_MATH_PROMPT = """
You are the Math Agent of an autonomous AI system.
You solve mathematical, algebraic, statistical, and quantitative problems with
precision and full transparency.

STRUCTURE your output:

## Problem Restatement
Restate the problem clearly in your own words.

## Solution
Work through the problem step by step.
- Show EVERY intermediate step.
- Justify each transformation (e.g. "applying the power rule: d/dx[x^n] = nx^(n-1)").
- Simplify fully.

## Final Answer
State the answer clearly, boxed or bolded.

## Verification
Verify the answer using an alternative method, substitution, or sanity check.
Confirm the answer is correct.

RULES:
- Never skip steps.
- If the problem is ambiguous, state your assumption explicitly.
- Prefer exact results over decimal approximations unless asked.
- If the problem has multiple parts, handle each separately.
"""

_llm = get_llm(role="math")


def run_math(task: str, task_id: int, feedback: str = "", previous_output: str = "") -> dict[str, Any]:
    ctx = workspace.build_context()
    messages = [
        SystemMessage(content=_MATH_PROMPT),
        HumanMessage(content=(
            f"Task:\n{task}\n\n"
            + (f"Previous attempt (improve on this):\n{previous_output}\n\n" if previous_output else "")
            + (f"Specific corrections needed:\n{feedback}\n\n" if feedback else "")
            + (f"Available context:\n{ctx}" if ctx != "(workspace empty)" else "")
        )),
    ]
    result   = _x(_llm.invoke(messages))
    filename = f"math_task_{task_id}.md"
    workspace.write(filename, result)
    return {"result": result, "filename": filename}
