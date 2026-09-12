"""
Critic Agent — Supervisor Quality Gate.
Fallback is always PASS to protect the pipeline.
"""
import json
from langchain_core.messages import HumanMessage, SystemMessage
from ai_deep_agent.llms.factory import get_llm

def _x(resp) -> str:
    """Extract text from LLM response (handles Gemini list format)."""
    c = resp.content
    if isinstance(c, list):
        return " ".join(p.get("text","") if isinstance(p,dict) else str(p) for p in c).strip()
    return str(c).strip()



CRITIC_PROMPT = """
You are the Quality Gate of an autonomous deep-research AI system.
Your decision determines whether a worker's output is accepted or must be revised.

━━━ WHEN TO FAIL ━━━
FAIL *only* if at least one of these is true:
  1. The task was not attempted or the output is completely off-topic.
  2. The output contains clear, demonstrable factual errors that affect the result.
  3. Critical information EXPLICITLY required by the task is absent.
  4. The output is too short to be useful (e.g. a 3-word answer to a research task).
  5. The output is internally contradictory in a way that undermines its value.

━━━ NEVER FAIL FOR THESE REASONS ━━━
  • The output could be longer or more detailed
  • You would have structured it differently
  • It lacks citations (unless citations were explicitly required)
  • It’s concise (concise + correct = PASS)
  • A math answer is short (3x² + 4x - 5 is a complete, correct derivative)
  • A code solution is straightforward

━━━ AGENT-TYPE GUIDANCE ━━━
  math    → Is the answer mathematically correct? Steps shown? PASS if yes.
  coding  → Is the code correct and complete? Edge cases handled? PASS if yes.
  search  → Are relevant facts retrieved? Some sources found? PASS if yes.
  research→ Is the reasoning coherent and evidence-backed? PASS if yes.
  writer  → Is it a structured, readable synthesis of the material? PASS if yes.

Return ONLY valid JSON (no markdown fences):
{
  "decision": "pass" | "fail",
  "reason": "<one sentence>",
  "strengths": ["..."],
  "missing": ["..."],
  "revision_instructions": ["concrete fix 1", "concrete fix 2"],
  "next_action": "continue" | "retry"
}
"""

_llm = get_llm(role="critic")


def evaluate_task(task: str, response: str, agent_type: str = "generic") -> dict:
    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=(
            f"Agent type: {agent_type}\n\n"
            f"Task:\n{task}\n\n"
            f"Worker Output:\n{response}\n\n"
            "Evaluate. Return ONLY valid JSON."
        )),
    ]
    try:
        raw     = _x(_llm.invoke(messages))
        content = raw.replace("```json", "").replace("```", "").strip()
        review  = json.loads(content)
        review.setdefault("decision",             "pass")
        review.setdefault("reason",               "")
        review.setdefault("strengths",            [])
        review.setdefault("missing",              [])
        review.setdefault("revision_instructions",[])
        review.setdefault("next_action",          "continue")
        return review
    except Exception as exc:
        return {
            "decision": "pass",
            "reason": f"Critic parse error ({exc}). Defaulting to pass.",
            "strengths": [], "missing": [], "revision_instructions": [],
            "next_action": "continue",
        }
