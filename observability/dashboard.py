import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from observability.audit_logger import get_observability_summary, read_audit_logs

console = Console()


def render_dashboard():
    summary = get_observability_summary()
    logs = read_audit_logs(limit=15)

    console.print()
    console.print("[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]            KnowledgeGuard — Observability & Audit Dashboard      [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print()

    # KPI Summary Cards
    kpi_table = Table.grid(expand=True)
    kpi_table.add_column(justify="center")
    kpi_table.add_column(justify="center")
    kpi_table.add_column(justify="center")
    kpi_table.add_column(justify="center")

    total_q = summary["total_queries"]
    denial_rate = f"{summary['access_denial_rate']}%"
    refusal_rate = f"{summary['refusal_rate']}%"
    avg_lat = f"{summary['avg_latency_ms']} ms"

    p1 = Panel(f"[bold green]{total_q}[/bold green]\nTotal Queries", border_style="green")
    p2 = Panel(f"[bold yellow]{denial_rate}[/bold yellow]\nAccess Denials", border_style="yellow")
    p3 = Panel(f"[bold blue]{refusal_rate}[/bold blue]\nRefusal Responses", border_style="blue")
    p4 = Panel(f"[bold magenta]{avg_lat}[/bold magenta]\nAvg Latency", border_style="magenta")

    kpi_table.add_row(p1, p2, p3, p4)
    console.print(kpi_table)
    console.print()

    # Recent Audit Log Table
    table = Table(title="Recent Query & Access Audit Log", border_style="cyan")
    table.add_column("Time (UTC)", style="dim", width=12)
    table.add_column("User", style="bold yellow", width=8)
    table.add_column("Decision", width=14)
    table.add_column("Question", style="white", min_width=30)
    table.add_column("Retrieved", justify="right", width=9)
    table.add_column("Denied", justify="right", width=8)
    table.add_column("Latency", justify="right", width=10)

    if not logs:
        console.print("[dim]No query events logged yet in logs/audit.jsonl.[/dim]")
    else:
        for log in logs:
            time_str = log.get("timestamp", "").split("T")[-1][:8]
            user = log.get("user_id", "")
            decision = log.get("access_decision", "")
            if decision == "GRANTED":
                dec_styled = "[green]GRANTED[/green]"
            elif decision == "PARTIAL_ACCESS":
                dec_styled = "[yellow]PARTIAL[/yellow]"
            elif decision == "DENIED":
                dec_styled = "[bold red]DENIED[/bold red]"
            else:
                dec_styled = "[dim]NO_MATCH[/dim]"

            question = log.get("question", "")
            if len(question) > 35:
                question = question[:32] + "..."

            ret = log.get("retrieval", {})
            auth_c = str(ret.get("authorized_count", 0))
            denied_c = str(ret.get("denied_count", 0))
            lat = f"{log.get('latency_ms', 0):.0f} ms" if log.get("latency_ms") else "-"

            table.add_row(time_str, user, dec_styled, question, auth_c, denied_c, lat)

        console.print(table)
        console.print()

    # Security & Access Breakdown
    sec_table = Table(title="Security & Permission Insights", border_style="magenta")
    sec_table.add_column("Top Authorized Documents", style="green", ratio=1)
    sec_table.add_column("Top Denied / Blocked Documents", style="red", ratio=1)

    top_auth = [f"{src} ({cnt})" for src, cnt in summary["top_authorized_docs"]] or ["None"]
    top_denied = [f"{src} ({cnt})" for src, cnt in summary["top_denied_docs"]] or ["None"]

    max_len = max(len(top_auth), len(top_denied))
    for i in range(max_len):
        a_str = top_auth[i] if i < len(top_auth) else ""
        d_str = top_denied[i] if i < len(top_denied) else ""
        sec_table.add_row(a_str, d_str)

    console.print(sec_table)
    console.print()


if __name__ == "__main__":
    render_dashboard()

