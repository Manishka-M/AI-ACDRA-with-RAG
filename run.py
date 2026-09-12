"""
AI Deep-Agent v3  —  Full System Runner
========================================
Runs the complete autonomous agent pipeline and prints EVERY step:
  • Planner task decomposition
  • Worker execution (search / research / math / coding / writer)
  • Critic quality evaluation
  • Retry loops with critique feedback
  • Final answer (full markdown)
  • Sources collected
  • Execution summary table

Usage:
    python run.py
    python run.py --query "Your query here"
    python run.py --query "Your query" --save
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ---- Rich ----
from rich.console import Console
from rich.prompt  import Prompt
from rich.panel   import Panel
from rich.rule    import Rule
console = Console()

# ---- Load .env before anything else ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---- Check API keys ----
def _check_env() -> bool:
    missing = [k for k in ["GROQ_API_KEY", "TAVILY_API_KEY"]
               if not os.getenv(k, "").strip()]
    if missing:
        console.print(Panel(
            "[bold red]Missing API keys in your .env file:[/bold red]\n"
            + "\n".join(f"  • {k}" for k in missing)
            + "\n\n[dim]Copy .env.example → .env and fill in your keys.[/dim]",
            border_style="red",
        ))
        return False
    return True


EXAMPLE_QUERIES = [
    "How has India asserted its geopolitical influence in the recent decade? What factors led to the rise of Indian diplomacy?",
    "Compare Rust and Go for building high-throughput microservices. Which is better and why?",
    "Differentiate x³ + 2x² − 5x + 7 and explain every step using the power rule.",
    "Implement binary search in Java with full complexity analysis and edge-case handling.",
    "What is the current state of AI regulation globally? Compare the EU AI Act vs US approaches.",
]


def _pick_query_interactive() -> str:
    console.print()
    console.print("[bold cyan]Example queries:[/bold cyan]")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        console.print(f"  [dim]{i}.[/dim] {q[:90]}{'...' if len(q) > 90 else ''}")
    console.print()
    choice = Prompt.ask(
        "[bold yellow]Enter your query (or press 1–5 for an example)[/bold yellow]",
        default="1",
    ).strip()
    if choice.isdigit() and 1 <= int(choice) <= len(EXAMPLE_QUERIES):
        return EXAMPLE_QUERIES[int(choice) - 1]
    return choice


def run(query: str, save: bool = False) -> dict:
    """
    Run the full AI Deep-Agent pipeline on *query*.
    Every agent step is printed live to the terminal via the rich UI.
    Returns the final result dict.
    """
    from ai_deep_agent.graph.builder import app
    from ai_deep_agent.state.state   import AgentState
    import ai_deep_agent.display.console as ui

    # Show header
    ui.header(query)

    # Build initial state
    initial: AgentState = {
        "messages":          [],
        "user_query":        query,
        "todos":             [],
        "planning_complete": False,
        "completed_tasks":   [],
        "final_answer":      "",
        "writer_used":       False,
        "retry_counts":      {},
        "execution_log":     [],
        "sources":           [],
    }

    t0     = time.time()
    result = app.invoke(initial)
    dur    = time.time() - t0

    # ---- Print final answer ----
    ui.final_answer(result["final_answer"], result.get("writer_used", False))

    # ---- Print sources ----
    ui.sources_panel(result.get("sources", []))

    # ---- Print execution summary table ----
    ui.summary_table(result.get("completed_tasks", []), dur)

    # ---- Save output to file? ----
    if save:
        out_dir  = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        ts       = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"run_{ts}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Query\n{query}\n\n")
            f.write(f"# Final Answer\n\n{result['final_answer']}\n\n")
            if result.get("sources"):
                f.write("# Sources\n")
                for s in result["sources"]:
                    f.write(f"- [{s.get('title','')}]({s.get('url','')})\n")
            f.write(f"\n---\n*Duration: {dur:.1f}s*\n")
        console.print(f"  [dim green]✓ Output saved to:[/dim green] {out_path}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AI Deep-Agent v3 — Autonomous Cognitive Research Engine"
    )
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Query to run (interactive picker if omitted)")
    parser.add_argument("--save",  "-s", action="store_true",
                        help="Save final answer to output/run_<timestamp>.md")
    args = parser.parse_args()

    if not _check_env():
        sys.exit(1)

    query = args.query.strip() if args.query else _pick_query_interactive()
    if not query:
        console.print("[red]No query provided. Exiting.[/red]")
        sys.exit(1)

    run(query=query, save=args.save)


if __name__ == "__main__":
    main()
