"""
Rich terminal display — shows the FULL output of every system step.
"""
from __future__ import annotations
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console(highlight=False)

WORKER_STYLES = {
    "search":   ("cyan",          "Search Agent   🔍"),
    "research": ("magenta",       "Research Agent 🧠"),
    "math":     ("yellow",        "Math Agent     📏"),
    "coding":   ("green",         "Coding Agent   💻"),
    "writer":   ("bright_magenta","Writer Agent   ✍️"),
    "default":  ("white",         "Agent          ⚙️"),
}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def header(query: str) -> None:
    console.print()
    console.print(Rule(style="bright_blue"))
    console.print(
        Panel(
            f"[bold bright_white]AI-AUTONOMOUS COGNITIVE ENGINE[/bold bright_white]  —  [dim]FOR DEEP-RESEARCH AND LONG HORIZON TASKS[/dim]\n\n"
            f"[bold yellow]Query:[/bold yellow]  {query}",
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print(Rule(style="bright_blue"))
    console.print()


def planner_done(todos: list[dict]) -> None:
    console.print(Panel(
        f"[bold green]✓ Planner created {len(todos)} task(s)[/bold green]",
        border_style="green", padding=(0, 2),
    ))
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    t.add_column("#",    style="dim",          width=4)
    t.add_column("Task", style="bright_white",  no_wrap=False)
    for todo in todos:
        t.add_row(str(todo["id"]), todo["task"])
    console.print(t)
    console.print()


def task_start(task_id: int, total: int, task: str, worker: str) -> None:
    color, label = WORKER_STYLES.get(worker, WORKER_STYLES["default"])
    console.print(Rule(
        f"[bold {color}]Task {task_id}/{total}  →  {label}[/bold {color}]",
        style=color,
    ))
    console.print(f"  [dim]{_ts()}[/dim]  [italic bright_white]{task}[/italic bright_white]")
    console.print()


def worker_result(worker: str, result: str, attempt: int) -> None:
    color, label = WORKER_STYLES.get(worker, WORKER_STYLES["default"])
    attempt_tag  = "" if attempt == 0 else f"  [dim](attempt #{attempt + 1})[/dim]"
    console.print(Panel(
        Markdown(result),
        title=f"[bold {color}]{label} Output[/bold {color}]{attempt_tag}",
        border_style=color,
        padding=(1, 2),
    ))
    console.print()


def critic_pass(reason: str, strengths: list[str]) -> None:
    body = f"[bold green]✔ PASS[/bold green]  —  {reason}"
    if strengths:
        body += "\n" + "\n".join(f"  [dim green]+ {s}[/dim green]" for s in strengths)
    console.print(Panel(body, title="[bold green]Critic Evaluation[/bold green]",
                        border_style="green", padding=(0, 2)))
    console.print()


def critic_fail(reason: str, missing: list[str], instructions: list[str]) -> None:
    body  = f"[bold red]✘ FAIL[/bold red]  —  {reason}\n"
    body += "\n".join(f"  [red]• Missing:[/red] {m}" for m in missing)
    body += "\n".join(f"  [yellow]• Fix:[/yellow] {i}" for i in instructions)
    console.print(Panel(body, title="[bold red]Critic Evaluation[/bold red]",
                        border_style="red", padding=(0, 2)))
    console.print()


def retry_notice(attempt: int, max_retries: int, feedback: str) -> None:
    console.print(Panel(
        f"[bold yellow]🔄 Retry {attempt}/{max_retries}[/bold yellow]\n"
        f"[dim]Feedback sent to worker:[/dim] {feedback[:200]}",
        border_style="yellow", padding=(0, 2),
    ))
    console.print()


def max_retries_hit(task_id: int) -> None:
    console.print(f"  [dim red]Max retries reached for task {task_id}. Accepting best result.[/dim red]\n")


def task_accepted(task_id: int, worker: str, retries: int) -> None:
    tag = f"  (after {retries} retr{'y' if retries == 1 else 'ies'})" if retries else ""
    console.print(f"  [green]✓[/green] Task {task_id} complete  [{worker}]{tag}\n")


def sources_panel(sources: list[dict]) -> None:
    if not sources:
        return
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=False)
    t.add_column("#",     width=4, style="dim")
    t.add_column("Title", style="bright_white", no_wrap=False)
    t.add_column("URL",   style="blue underline", no_wrap=True)
    for i, s in enumerate(sources, 1):
        t.add_row(str(i), s.get("title", "")[:60], s.get("url", "")[:80])
    console.print(Panel(t, title="[bold cyan]Sources Collected[/bold cyan]",
                        border_style="cyan", padding=(0, 1)))
    console.print()


def final_answer(answer: str, writer_used: bool) -> None:
    label  = "✍️  Final Report (Writer synthesised)" if writer_used else "📌  Final Answer"
    console.print(Rule(style="bright_blue"))
    console.print(Panel(
        Markdown(answer),
        title=f"[bold bright_white]{label}[/bold bright_white]",
        border_style="bright_blue",
        padding=(1, 3),
    ))
    console.print(Rule(style="bright_blue"))
    console.print()


def summary_table(completed: list[dict], duration: float) -> None:
    t = Table(
        title="Execution Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    t.add_column("#",       width=4,  style="dim")
    t.add_column("Worker",  width=12, style="cyan")
    t.add_column("Decision",width=10)
    t.add_column("Retries", width=9,  style="yellow")
    t.add_column("Task",    style="bright_white", no_wrap=False)
    for task in completed:
        dec  = task.get("review", {}) or {}
        d    = dec.get("decision", "pass")
        dcol = "green" if d == "pass" else "red"
        t.add_row(
            str(task["id"]),
            task["worker"].upper(),
            f"[{dcol}]{d.upper()}[/{dcol}]",
            str(task.get("retries", 0)),
            task["task"][:70],
        )
    console.print(t)
    console.print(f"\n  [dim]Total duration: {duration:.1f}s[/dim]\n")
