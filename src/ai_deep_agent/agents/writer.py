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



_WRITER_PROMPT = """
You are the Writer Agent of an autonomous deep-research AI system.
You produce the polished, user-facing final report.

You have access to all prior research, search results, and analyses
collected throughout the workflow. Your job is to synthesize them into
a coherent, professional document.

━━━ REPORT STRUCTURE ━━━

# [Compelling Report Title]

## Executive Summary
3-5 sentences. The most important findings and conclusions.
A busy reader should understand the entire report from this section alone.

## Introduction
Context and framing. Why does this topic matter? What question is being answered?

## [Section 1: Empirical Findings / Data]
Key facts, statistics, events, timelines.
Use tables where data is comparative.

## [Section 2: Analysis]
Deeper reasoning over the evidence. Patterns, cause-effect chains,
competing interpretations. This is your analytical core.

## [Section 3: Case Studies / Examples]
2-3 concrete, specific examples that illustrate the key points.
Do not be vague — name people, places, dates, outcomes.

## [Additional sections as appropriate]

## Conclusions
What can be definitively concluded from this research?
What remains uncertain? What are the implications?

## References & Sources
List all sources used:
- [Title](URL)

━━━ QUALITY RULES ━━━
  1. NEVER invent facts, statistics, quotes, or events.
  2. Every analytical claim must be grounded in the research material.
  3. Write in a clear, authoritative, third-person voice.
  4. Use headings, sub-headings, bullets, and tables to aid readability.
  5. This is a FINAL document — it must stand alone without the workflow context.
  6. Aim for depth + clarity. Not length for its own sake.
  7. Include all relevant URLs in the References section.
"""

_llm = get_llm(role="writer")

def run_writer(task: str, task_id: int, feedback: str = "", previous_output: str = "") -> dict[str, Any]:
    ctx = workspace.build_context(exclude=["final_report.md"])
    messages = [
        SystemMessage(content=_WRITER_PROMPT),
        HumanMessage(content=(
            f"Report Task:\n{task}\n\n"
            + (f"Previous draft (improve on this):\n{previous_output}\n\n" if previous_output else "")
            + (f"Revision instructions:\n{feedback}\n\n" if feedback else "")
            + f"All Research Material:\n{ctx}"
        )),
    ]
    result   = _x(_llm.invoke(messages))
    workspace.write("final_report.md", result)
    return {"result": result, "filename": "final_report.md", "writer_used": True}
