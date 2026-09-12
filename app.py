"""
AI_AUTONOMOUS COGNITIVE ENGINE FOR DEEP-RESEARCH AND LONG HORIZON TASKS
Streamlit Web Interface  —  v5 (clean output: final report only)
"""
import os
import sys
import time
import streamlit as st

# Add src/ to path so ai_deep_agent package is found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ------------------------------------------------------------------ #
#  Page config  (must be first Streamlit call)                        #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="AI Autonomous Cognitive Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Load secrets / env                                                 #
# ------------------------------------------------------------------ #
def _load_secrets():
    """Pull keys from Streamlit Cloud secrets or fall back to .env file."""
    for key in ["ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "GEMINI_API_KEY",
                "GEMINI_MODEL", "GEMINI_ROLES", "TAVILY_API_KEY"]:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]

try:
    _load_secrets()
except Exception:
    pass   # running locally without secrets

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------ #
#  Sidebar                                                            #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4616/4616013.png", width=64)
    st.title("🧠 AI Cognitive Engine")
    st.caption("Deep-Research · Long Horizon Tasks · Autonomous Multi-Agent System")
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
| ⚖️ Critic | Quality gate on final report |
""")
    st.divider()
    st.markdown("### ⚙️ Config")
    model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("GEMINI_MODEL", "not set")
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
#  Run agent and display results                                      #
# ------------------------------------------------------------------ #
if run_btn and query.strip():
    # Check required keys
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

    initial: AgentState = {
        "messages":           [],
        "user_query":         query.strip(),
        "todos":              [],
        "planning_complete":  False,
        "completed_tasks":    [],
        "final_answer":       "",
        "writer_used":        False,
        "retry_counts":       {},
        "execution_log":      [],
        "sources":            [],
    }

    with st.spinner("🧠 Agent thinking..."):
        start   = time.time()
        result  = app.invoke(initial)
        elapsed = time.time() - start

    # ------------------------------------------------------------------ #
    #  Display results                                                    #
    # ------------------------------------------------------------------ #
    st.success(f"✅ Completed in {elapsed:.1f}s")
    st.divider()

    # --- Task execution summary (compact, no full worker outputs) ---
    completed_tasks = result.get("completed_tasks", [])
    if completed_tasks:
        with st.expander("📋 Task Execution Summary", expanded=False):
            worker_icons = {
                "search":   "🔍",
                "research": "🔬",
                "math":     "📊",
                "coding":   "💻",
                "writer":   "✍️",
            }
            for task in completed_tasks:
                worker  = task.get("worker", "")
                icon    = worker_icons.get(worker, "⚙️")
                task_id = task.get("id", "?")
                task_t  = task.get("task", "")[:80]
                review  = task.get("review") or {}
                verdict = review.get("decision", "pass")
                badge   = "✅" if verdict == "pass" else "🔄"
                retries = task.get("retries", 0)
                retry_tag = f" ({retries} retr{'y' if retries==1 else 'ies'})" if retries else ""
                st.markdown(f"{badge} **Task {task_id}** {icon} `{worker.capitalize()}`{retry_tag} — {task_t}...")

    # --- Final Report (beautiful, full width) ---
    # supervisor stores final output under 'final_answer'
    final_report = result.get("final_answer", "")

    st.markdown("## 📝 Final Report")
    st.divider()

    if final_report:
        st.markdown(final_report)
    else:
        # Fallback: last completed task result
        if completed_tasks:
            st.markdown(completed_tasks[-1].get("result", "*No output generated.*"))
        else:
            st.warning("⚠️ No output was generated. Try a simpler query or check your API keys.")

    # --- Sources ---
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

    # --- Download report ---
    st.divider()
    report_text = final_report or (completed_tasks[-1].get("result", "") if completed_tasks else "")
    if report_text:
        st.download_button(
            label="⬇️ Download Report",
            data=report_text,
            file_name="ai_research_report.md",
            mime="text/markdown",
        )
