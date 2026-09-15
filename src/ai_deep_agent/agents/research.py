"""Research Agent — deep analytical reasoning over workspace evidence.
v4-rag: RAG context injection added.
"""
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

MAX_CTX_CHARS = 5_000

_RESEARCH_PROMPT = """You are the Research Agent of an autonomous deep-research AI system.
You are a rigorous analytical thinker — part academic researcher, part investigative journalist.
You will be given workspace material (search results, prior analyses) and a task.
Produce a deep, evidence-grounded analytical document.

STRUCTURE:
## Executive Summary
2-3 sentences capturing the most important insight.

## Evidence Base
Key facts, data points, and evidence you are working from.
Be precise: numbers, dates, names, sources where available.

## Analysis
Organise by theme or argument:
### [Theme 1]
  - What the evidence shows
  - Why it matters
  - Counter-arguments or qualifications
### [Theme 2] ... (repeat)

## Synthesis & Key Insights
Patterns across evidence. Strongest conclusions. Uncertainties.

## Implications
What does this mean in practice?

QUALITY RULES:
  1. Every claim must be traceable to the evidence provided.
  2. Distinguish strong evidence from inference.
  3. Flag contradictions between sources.
  4. Never fabricate data, statistics, quotes, or events."""

_llm = get_llm(role="research")

def run_research(
    task: str,
    task_id: int,
    feedback: str = "",
    previous_output: str = "",
    rag_context: str = "",        # ← NEW: RAG se aaya context
) -> dict[str, Any]:

    ctx          = workspace.build_context()[:MAX_CTX_CHARS]
    prev_snippet = previous_output[:400] if previous_output else ""
    fb_snippet   = feedback[:300]        if feedback        else ""

    # ← NEW: RAG section build karo
    rag_section = ""
    if rag_context.strip():
        rag_section = (
            f"📚 USER-PROVIDED DOCUMENTS (use as primary source):\n"
            f"{rag_context[:2000]}\n\n"
        )

    messages = [
        SystemMessage(content=_RESEARCH_PROMPT),
        HumanMessage(content=(
            f"Task:\n{task}\n\n"
            + (f"Previous attempt (improve on this):\n{prev_snippet}\n\n" if prev_snippet else "")
            + (f"Gaps to address:\n{fb_snippet}\n\n"                      if fb_snippet  else "")
            + rag_section                                                   # ← NEW
            + f"Workspace Material:\n{ctx}"
        )),
    ]

    result   = _x(_llm.invoke(messages))
    filename = f"research_task_{task_id}.md"
    workspace.write(filename, result)
    return {"result": result, "filename": filename}
