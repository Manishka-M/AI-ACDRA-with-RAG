"""
Search Agent — multi-query, evidence-rich web search.
v3-fix: truncate raw search content to stay within Groq free-tier TPM limits.
"""
import re
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from ai_deep_agent.config.settings import TAVILY_API_KEY
from ai_deep_agent.llms.factory import get_llm
from ai_deep_agent.memory.virtual_fs import workspace

def _x(resp) -> str:
    """Extract text from LLM response (handles Gemini list format)."""
    c = resp.content
    if isinstance(c, list):
        return " ".join(p.get("text","") if isinstance(p,dict) else str(p) for p in c).strip()
    return str(c).strip()



# ---------------------------------------------------------------
# Token budget: keep total prompt under ~3,500 words (~4,500 tokens)
# to stay safely within Groq free-tier limits.
# ---------------------------------------------------------------
MAX_CONTENT_CHARS = 8_000   # raw search text cap
MAX_RESULT_CHARS  = 600     # per individual search result

_SEARCH_PROMPT = """
You are the Search Agent of an autonomous deep-research AI system.
You have been given raw search results.

Produce a well-structured Markdown document:

## Summary
3-5 sentences covering the key findings.

## Key Facts & Evidence
Bullet the most important data points, statistics, events, names, and dates.
Be specific — vague statements add no value.

## Detailed Findings
Organise by theme. Use sub-headings. Quote sources where compelling.

## Gaps & Uncertainties
Note what the search did NOT find or what remains unclear.

## Sources
- [Title](URL)

Rules: never fabricate facts. If two sources contradict, present both.
"""

_QUERY_GEN_PROMPT = """
Given a research task, return ONLY a Python list of 2 focused search queries
that together cover different angles. Example:
["query one", "query two"]
"""

_llm       = get_llm(role="search")
_query_llm = get_llm(role="search")
_client    = TavilyClient(api_key=TAVILY_API_KEY)


def _generate_queries(task: str, feedback: str = "") -> list[str]:
    prompt = f"Task: {task}" + (f"\nFeedback: {feedback}" if feedback else "")
    try:
        raw = _x(_query_llm.invoke([
            SystemMessage(content=_QUERY_GEN_PROMPT),
            HumanMessage(content=prompt),
        ])).replace("```python","").replace("```","").strip()
        queries = eval(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:2]   # max 2 queries to stay within rate limits
    except Exception:
        pass
    return [re.sub(r"^[\d\.\-\)]+\s*", "", task).strip()]


def _run_query(query: str) -> tuple[str, list[dict]]:
    try:
        results = _client.search(query=query, max_results=5, include_answer=True)
    except Exception as exc:
        return f"Search failed: {exc}", []

    parts, sources = [f"### Query: {query}\n"], []
    if results.get("answer"):
        parts.append(f"**Direct Answer:** {results['answer']}\n")
    for r in results.get("results", []):
        title   = r.get("title", "Untitled")
        url     = r.get("url", "")
        content = r.get("content", "")[:MAX_RESULT_CHARS]   # truncate per result
        parts.append(f"**{title}**\n{content}\n")
        if url:
            sources.append({"title": title, "url": url})
    return "\n".join(parts), sources


def run_search(
    task: str,
    task_id: int,
    feedback: str = "",
    previous_output: str = "",
) -> dict[str, Any]:
    queries = _generate_queries(task, feedback)

    all_text, all_sources = [], []
    for q in queries:
        text, srcs = _run_query(q)
        all_text.extend([text])
        all_sources.extend(srcs)

    # Hard cap on total raw text sent to LLM
    combined_raw = "\n\n".join(all_text)[:MAX_CONTENT_CHARS]

    # Keep previous_output short to save tokens
    prev_snippet = previous_output[:500] if previous_output else ""
    fb_snippet   = feedback[:300]         if feedback         else ""

    messages = [
        SystemMessage(content=_SEARCH_PROMPT),
        HumanMessage(content=(
            f"Task:\n{task}\n\n"
            + (f"Previous attempt (improve on this):\n{prev_snippet}\n\n" if prev_snippet else "")
            + (f"Gaps to address:\n{fb_snippet}\n\n"                       if fb_snippet   else "")
            + f"Raw Search Results:\n{combined_raw}"
        )),
    ]
    result   = _x(_llm.invoke(messages))
    filename = f"search_task_{task_id}.md"
    workspace.write(filename, result)
    return {"result": result, "sources": all_sources, "filename": filename}
