"""
Supervisor — the full cognitive engine.
Orchestrates planning, routing, execution, then a dedicated
writer + critic + retry loop for the final report.
"""
from __future__ import annotations
import re
from langchain_core.messages import HumanMessage, SystemMessage
from ai_deep_agent.agents.critic   import evaluate_task
from ai_deep_agent.agents.planner  import run_planner
from ai_deep_agent.agents.search   import run_search
from ai_deep_agent.agents.research import run_research
from ai_deep_agent.agents.math     import run_math
from ai_deep_agent.agents.coding   import run_coding
from ai_deep_agent.agents.writer   import run_writer
from ai_deep_agent.llms.factory    import get_llm
from ai_deep_agent.memory.virtual_fs import workspace
from ai_deep_agent.state.state     import AgentState
import ai_deep_agent.display.console as ui

MAX_RETRIES = 2

ROUTER_PROMPT = """
You are the Supervisor Router of an autonomous deep-research AI system.

Choose the single best worker for the assigned task:

  search   – retrieve current information, real-world facts, events, data from the web
  research – analyse, synthesise, compare, and reason over already-gathered information
  math     – solve mathematical, algebraic, statistical, or quantitative problems
  coding   – write, design, debug, or explain code, algorithms, implementations

Decision guide:
  • If the task requires FINDING information → search
  • If the task is to ANALYSE/COMPARE/REASON → research
  • If the task involves NUMBERS, ALGEBRA, STATISTICS → math
  • If the task involves CODE, ALGORITHMS, SOFTWARE → coding

Return ONLY one word: search | research | math | coding
No punctuation, no explanation.
"""

_router_llm = get_llm(role="supervisor")

DISPATCH = {
    "search":   run_search,
    "research": run_research,
    "math":     run_math,
    "coding":   run_coding,
}


def _clean(text: str) -> str:
    """Strip tool-call leak artifacts from small model outputs."""
    text = re.sub(r'<\|tool_call_start\|>.*?<\|tool_call_end\|>', '', text, flags=re.DOTALL)
    return text.strip()


def _route(task: str, ctx_summary: str = "") -> str:
    """LLM-based semantic routing with keyword fallback."""
    msg = task
    if ctx_summary:
        msg += f"\n\nWorkspace so far: {ctx_summary[:300]}"
    try:
        response = _router_llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"Task:\n{msg}"),
        ])
        c = response.content
        text = (
            " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c).strip()
            if isinstance(c, list) else str(c).strip()
        )
        text = _clean(text)
        worker = text.lower().strip().split()[0].rstrip(".,:") if text.split() else ""
        if worker in DISPATCH:
            return worker
    except Exception:
        pass

    # Keyword fallback for weak models
    tl = task.lower()
    if any(k in tl for k in ["search", "find", "retrieve", "current", "latest", "news", "look up", "real-world"]):
        return "search"
    if any(k in tl for k in ["code", "implement", "program", "algorithm", "function", "debug", "java", "python", "software"]):
        return "coding"
    if any(k in tl for k in ["calculat", "differentiat", "integrat", "equation", "solve", "proof", "statistic", "math"]):
        return "math"
    return "research"


def run_supervisor(state: AgentState) -> AgentState:
    log:     list[dict] = state.get("execution_log", [])
    sources: list[dict] = state.get("sources", [])
    retries: dict       = state.get("retry_counts", {})

    # ──────────────────── PLANNING ────────────────────
    if not state.get("planning_complete", False):
        workspace.clear()
        todos = run_planner(state["user_query"])
        state.update({
            "todos":             todos,
            "planning_complete": True,
            "completed_tasks":   [],
            "writer_used":       False,
        })
        ui.planner_done(todos)
        log.append({"event": "planner_done", "tasks": [t["task"] for t in todos]})

    todos           = state.get("todos", [])
    completed_tasks = state.get("completed_tasks", [])
    completed_ids   = {t["id"] for t in completed_tasks}

    # ──────────────────── EXECUTE RESEARCH TASKS ────────────────────
    # Workers: search / research / math / coding
    # Writer is NOT dispatched here — it always runs separately at the end.
    for todo in todos:
        if todo["id"] in completed_ids:
            continue

        task_id   = todo["id"]
        task_desc = todo["task"]
        ctx_sum   = workspace.build_context()[:200]
        worker    = _route(task_desc, ctx_sum)

        ui.task_start(task_id, len(todos), task_desc, worker)
        log.append({"event": "task_start", "id": task_id, "worker": worker, "task": task_desc})

        out    = DISPATCH[worker](
            task=task_desc, task_id=task_id,
            feedback="", previous_output="",
        )
        result = out["result"]

        # Accumulate sources
        for src in out.get("sources", []):
            if src not in sources:
                sources.append(src)

        retries_done = retries.get(task_id, 0)
        ui.task_accepted(task_id, worker, retries_done)
        completed_tasks.append({
            "id":      task_id,
            "task":    task_desc,
            "worker":  worker,
            "result":  result,
            "review":  {"decision": "pass", "reason": "auto-accepted"},
            "retries": 0,
        })
        log.append({"event": "task_complete", "id": task_id})

    # ──────────────────── FINAL REPORT: WRITER + CRITIC + RETRY ────────────────────
    writer_task = (
        f"Write a comprehensive, well-structured final report answering: {state['user_query']}\n"
        "Use all the research material gathered in the workspace."
    )

    attempt         = 0
    previous_output = ""
    feedback        = ""
    final_answer    = ""

    while attempt <= MAX_RETRIES:
        out    = run_writer(
            task=writer_task, task_id=9999,
            feedback=feedback, previous_output=previous_output,
        )
        result = _clean(out["result"])
        state["writer_used"] = True

        # Show writer output
        ui.worker_result("writer", result, attempt)

        # Critic evaluates
        review = evaluate_task(writer_task, result, agent_type="writer")
        log.append({"event": "critic", "id": 9999, "review": review})

        if review["decision"] == "pass":
            ui.critic_pass(review["reason"], review.get("strengths", []))
            final_answer = result
            break
        else:
            ui.critic_fail(
                review["reason"],
                review.get("missing", []),
                review.get("revision_instructions", []),
            )
            if attempt >= MAX_RETRIES:
                ui.max_retries_hit(9999)
                final_answer = result
                break
            feedback = "; ".join(
                review.get("revision_instructions", []) or [review.get("reason", "")]
            )
            previous_output = result
            attempt        += 1
            retries[9999]   = attempt
            ui.retry_notice(attempt, MAX_RETRIES, feedback)
            log.append({"event": "retry", "id": 9999, "attempt": attempt, "feedback": feedback})

    # ──────────────────── DONE ────────────────────
    state.update({
        "completed_tasks": completed_tasks,
        "final_answer":    final_answer,
        "execution_log":   log,
        "sources":         sources,
        "retry_counts":    retries,
    })
    return state
