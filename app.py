"""
AI AUTONOMOUS COGNITIVE ENGINE — Deep Research & Long Horizon Tasks
Streamlit Web Interface — v6
Features:
  - Live process display (planning, delegation, writer, critic, retry)
  - Tool-call artifact stripping from final report
  - Full OpenRouter support
"""
import os
import re
import sys
import time
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ------------------------------------------------------------------ #
#  Page config                                                        #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="AI Autonomous Cognitive Deep-Research Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Load secrets / env                                                 #
# ------------------------------------------------------------------ #
def _load_secrets():
    for key in [
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
        "GEMINI_API_KEY",    "GEMINI_MODEL", "GEMINI_ROLES",
        "OPENROUTER_API_KEY","OPENROUTER_MODEL",
        "TAVILY_API_KEY",
    ]:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]

try:
    _load_secrets()
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------ #
#  Helper: strip tool-call artifacts from model output               #
# ------------------------------------------------------------------ #
def _strip_artifacts(text: str) -> str:
    if not text:
        return text
    # Remove <|tool_call_start|>...<|tool_call_end|>
    text = re.sub(
        r'<\|tool_call_start\|>.*?<\|tool_call_end\|>',
        '', text, flags=re.DOTALL
    )
    # Remove [google(...)] style raw tool calls
    text = re.sub(r'\[(?:google|search|fetch)\(.*?\)(?:,\s*(?:google|search|fetch)\(.*?\))*\]', '', text, flags=re.DOTALL)
    return text.strip()

# ------------------------------------------------------------------ #
#  Sidebar                                                            #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4616/4616013.png", width=64)
    st.title("🧠 AI Autonomous Cognitive Deep-Research Engine")
    st.caption("Deep-Research · Long Horizon Tasks · Autonomous Multi-Agent")
    st.divider()
    st.markdown("### 🧠 Agent Architecture")
    st.markdown("""
| Agent | Role |
|---|---|
| 📌 Planner | Decomposes query into tasks |
| 🔍 Search | Live web search & evidence |
| 🔬 Research | Deep analysis & synthesis |
| 📊 Math | Calculations & proofs |
| 💻 Coding | Code generation & debug |
| ✍️ Writer | Final report synthesis |
| ⚖️ Critic | Quality gate + retry loop |
""")
    st.divider()
    model = (
        os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or "Kya Farak Padta Hai, Chal toh raha hai na?"
    )
    st.markdown("### ⚙️ Config")
    st.code(f"Model: {model}")
    st.divider()
    st.caption("👨‍💻 Built by Manishka | Placement Project")

# ------------------------------------------------------------------ #
#  Main UI                                                            #
# ------------------------------------------------------------------ #
st.markdown("""
<h1 style='text-align:center;'>
    🧠 AI Autonomous Cognitive Engine
</h1>
<p style='text-align:center; color:gray;'>
    Ask anything — the agent plans, researches, reasons, and writes a full report.
</p>
""", unsafe_allow_html=True)

st.divider()

query = st.text_area(
    "🔍 Enter your research query",
    placeholder="e.g. How will AI impact cybersecurity laws over the next decade?",
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🚀 Run Agent", type="primary", use_container_width=True)
with col2:
    st.markdown("*The agent will plan, research, and write a full structured report.*")

# ------------------------------------------------------------------ #
#  Run Agent                                                          #
# ------------------------------------------------------------------ #
if run_btn and query.strip():

    missing = [k for k in ["TAVILY_API_KEY"]
               if not os.environ.get(k, "").strip()]
    if missing:
        st.error(
            f"🔴 Missing API keys: {', '.join(missing)}\n\n"
            "Add them to Streamlit Cloud Secrets or your local .env file."
        )
        st.stop()

    from ai_deep_agent.graph.builder import app
    from ai_deep_agent.state.state   import AgentState
    import ai_deep_agent.display.console as ui

    # ------------------------------------------------------------------ #
    #  Live Process Display Setup                                        #
    # ------------------------------------------------------------------ #
    st.markdown("### ⚡ Live Agent Process")
    process_box = st.empty()   # updates in real-time
    live_log: list[str] = []

    ICONS = {
        "search":   "🔍",
        "research": "🔬",
        "math":     "📊",
        "coding":   "💻",
        "writer":   "✍️",
    }

    def _refresh():
        """Re-render live log inside the placeholder."""
        with process_box.container():
            for line in live_log:
                st.markdown(line, unsafe_allow_html=True)

    def _log(msg: str):
        live_log.append(msg)
        _refresh()

    # ------------------------------------------------------------------ #
    #  Monkey-patch ui.* so every terminal event also shows on screen    #
    # ------------------------------------------------------------------ #

    _orig_planner_done = ui.planner_done
    def _st_planner_done(todos):
        _orig_planner_done(todos)
        _log(f"")
        _log(f"**📌 Planning complete — {len(todos)} task(s) created:**")
        for t in todos:
            _log(f"&nbsp;&nbsp;&nbsp;&nbsp;`{t['id']}.` {t['task']}")
        _log("---")
    ui.planner_done = _st_planner_done

    _orig_task_start = ui.task_start
    def _st_task_start(task_id, total, task, worker):
        _orig_task_start(task_id, total, task, worker)
        icon = ICONS.get(worker, "⚙️")
        _log(f"{icon} **Task {task_id}/{total}** → `{worker.capitalize()} Agent` — *{task[:80]}...*")
    ui.task_start = _st_task_start

    _orig_task_accepted = ui.task_accepted
    def _st_task_accepted(task_id, worker, retries):
        _orig_task_accepted(task_id, worker, retries)
        _log(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ Task {task_id} complete")
    ui.task_accepted = _st_task_accepted

    _orig_worker_result = ui.worker_result
    def _st_worker_result(worker, result, attempt):
        _orig_worker_result(worker, result, attempt)
        label = f" (attempt #{attempt + 1})" if attempt > 0 else ""
        _log(f"✍️ **Writer Agent{label}** — synthesising final report...")
    ui.worker_result = _st_worker_result

    _orig_critic_pass = ui.critic_pass
    def _st_critic_pass(reason, strengths):
        _orig_critic_pass(reason, strengths)
        _log(f"✅ **Critic — PASS:** {reason[:120]}")
        for s in (strengths or [])[:3]:
            _log(f"&nbsp;&nbsp;&nbsp;&nbsp;✨ {s}")
    ui.critic_pass = _st_critic_pass

    _orig_critic_fail = ui.critic_fail
    def _st_critic_fail(reason, missing, instructions):
        _orig_critic_fail(reason, missing, instructions)
        _log(f"❌ **Critic — FAIL:** {reason[:120]}")
        for m in (missing or [])[:4]:
            _log(f"&nbsp;&nbsp;&nbsp;&nbsp;• Missing: {m}")
        for fix in (instructions or [])[:2]:
            _log(f"&nbsp;&nbsp;&nbsp;&nbsp;🔧 Fix: {fix[:100]}")
    ui.critic_fail = _st_critic_fail

    _orig_retry = ui.retry_notice
    def _st_retry(attempt, max_retries, feedback):
        _orig_retry(attempt, max_retries, feedback)
        _log(f"🔄 **Retry {attempt}/{max_retries}** — sending feedback to Writer...")
        _log(f"&nbsp;&nbsp;&nbsp;&nbsp;*{feedback[:150]}*")
    ui.retry_notice = _st_retry

    _orig_max_retries = ui.max_retries_hit
    def _st_max_retries(task_id):
        _orig_max_retries(task_id)
        _log("⚠️ **Max retries reached** — using best available output")
    ui.max_retries_hit = _st_max_retries

    # ------------------------------------------------------------------ #
    #  Kick off — show initial status                                    #
    # ------------------------------------------------------------------ #
    _log("🧠 **Agent started** — analysing your query...")
    _log(f"> *{query.strip()[:120]}*")
    _log("---")
    _log("⏳ **Planning** — breaking query into research tasks...")

    initial: AgentState = {
        "messages":          [],
        "user_query":        query.strip(),
        "todos":             [],
        "planning_complete": False,
        "completed_tasks":   [],
        "final_answer":      "",
        "writer_used":       False,
        "retry_counts":      {},
        "execution_log":     [],
        "sources":           [],
    }

    start  = time.time()
    result = app.invoke(initial)
    elapsed = time.time() - start

    _log("---")
    _log(f"🏁 **Done in {elapsed:.1f}s** — report ready below ↓")

    # ------------------------------------------------------------------ #
    #  Final Report                                                      #
    # ------------------------------------------------------------------ #
    st.divider()
    st.success(f"✅ Completed in {elapsed:.1f}s")
    st.markdown("## 📝 Final Report")
    st.divider()

    final_report = _strip_artifacts(result.get("final_answer", ""))
    completed    = result.get("completed_tasks", [])

    if final_report:
        st.markdown(final_report)
    elif completed:
        st.markdown(_strip_artifacts(completed[-1].get("result", "*No output generated.*")))
    else:
        st.warning("⚠️ No output was generated. Try a different query or check your API keys.")

    # ------------------------------------------------------------------ #
    #  Sources                                                           #
    # ------------------------------------------------------------------ #
    sources = result.get("sources", [])
    if sources:
        st.divider()
        with st.expander("🔗 Sources", expanded=False):
            seen = set()
            for s in sources:
                url = s.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    title = s.get("title", url)
                    st.markdown(f"- [{title}]({url})")

    # ------------------------------------------------------------------ #
    #  Execution Summary (collapsed)                                     #
    # ------------------------------------------------------------------ #
    if completed:
        with st.expander("📊 Execution Summary", expanded=False):
            worker_icons = {
                "search": "🔍", "research": "🔬",
                "math": "📊",   "coding": "💻", "writer": "✍️",
            }
            for task in completed:
                worker  = task.get("worker", "")
                icon    = worker_icons.get(worker, "⚙️")
                tid     = task.get("id", "?")
                tdesc   = task.get("task", "")[:80]
                verdict = (task.get("review") or {}).get("decision", "pass")
                badge   = "✅" if verdict == "pass" else "🔄"
                retries = task.get("retries", 0)
                r_tag   = f" ({retries} retr{'y' if retries==1 else 'ies'})" if retries else ""
                st.markdown(f"{badge} **Task {tid}** {icon} `{worker.capitalize()}`{r_tag} — {tdesc}...")

    # ------------------------------------------------------------------ #
    #  Download                                                          #
    # ------------------------------------------------------------------ #
    report_text = final_report or (completed[-1].get("result", "") if completed else "")
    if report_text:
        st.divider()
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=report_text,
            file_name="ai_research_report.md",
            mime="text/markdown",
        )
