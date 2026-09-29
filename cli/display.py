"""Rich-based rendering helpers for the command line interface."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

LEVEL_STYLE = {"Beginner": "green", "Intermediate": "yellow", "Advanced": "red"}


def banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]Personalized Learning Resource Recommender[/bold cyan]\n"
        "[dim]Hybrid AI recommender: Content-based + Collaborative + Knowledge-based[/dim]",
        border_style="cyan",
    ))


def level(lvl: str) -> str:
    return f"[{LEVEL_STYLE.get(lvl, 'white')}]{lvl}[/]"


def menu(title: str, options: list) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for key, label in options:
        table.add_row(f"[bold cyan]{key}[/]", label)
    console.print(Panel(table, title=f"[bold]{title}[/bold]", border_style="blue", expand=False))


def recommendations_table(recs: list, title: str = "Recommended for you") -> None:
    table = Table(title=title, show_lines=True, header_style="bold magenta")
    table.add_column("#", justify="right")
    table.add_column("ID", justify="right")
    table.add_column("Resource", min_width=24)
    table.add_column("Level")
    table.add_column("Format")
    table.add_column("Hours", justify="right")
    table.add_column("Cost")
    table.add_column("Score", justify="right")
    table.add_column("Why recommended?", min_width=30)
    for n, rec in enumerate(recs, 1):
        r = rec.resource
        table.add_row(
            str(n), str(r["id"]), f"[bold]{r['title']}[/]\n[dim]{r['topic']}[/]", level(r["level"]),
            r["format"], f"{r['duration_hours']:g}", r["cost"], f"{rec.score:.2f}",
            "\n".join(f"• {x}" for x in rec.reasons),
        )
    console.print(table)


def resources_table(items: list, title: str, score_label: str = None) -> None:
    """items: list of dicts or (dict, score) tuples."""
    table = Table(title=title, header_style="bold magenta")
    for col, kw in [("ID", {"justify": "right"}), ("Resource", {}), ("Topic", {}), ("Level", {}),
                    ("Format", {}), ("Hours", {"justify": "right"}), ("Cost", {}), ("Rating", {"justify": "right"})]:
        table.add_column(col, **kw)
    if score_label:
        table.add_column(score_label, justify="right")
    for it in items:
        r, s = it if isinstance(it, tuple) else (it, None)
        row = [str(r["id"]), r["title"], r["topic"], level(r["level"]), r["format"],
               f"{r['duration_hours']:g}", r["cost"], f"{r['rating']}"]
        if score_label:
            row.append(f"{s:.2f}")
        table.add_row(*row)
    console.print(table)


def resource_detail(r: dict, community: tuple, user_rating=None, completed=False, breakdown=None) -> None:
    avg, count = community
    lines = [
        f"[bold]{r['title']}[/bold]",
        f"[dim]{r['provider']}[/dim]\n",
        r["description"] + "\n",
        f"Topic: [cyan]{r['topic']}[/]   Level: {level(r['level'])}   Format: {r['format']}",
        f"Duration: {r['duration_hours']:g} h   Cost: {r['cost']}   Catalogue rating: {r['rating']}/5",
        f"Community: {avg if avg is not None else '-'} avg from {count} learner rating(s)",
        f"Tags: [dim]{r['tags'].replace(';', ', ')}[/dim]",
    ]
    if user_rating is not None:
        lines.append(f"Your rating: [yellow]{'★' * int(user_rating)}{'☆' * (5 - int(user_rating))}[/]")
    if completed:
        lines.append("[green]✔ Completed[/green]")
    if breakdown:
        parts = [f"{k}={v:.2f}" for k, v in breakdown.items() if v is not None]
        lines.append("\n[dim]Score breakdown: " + "  ".join(parts) + "[/dim]")
    console.print(Panel("\n".join(lines), title=f"Resource #{r['id']}", border_style="green"))


def learning_path(stages: list, goal: str) -> None:
    if not stages:
        console.print("[yellow]No matching resources found for that goal.[/yellow]")
        return
    total = sum(s["hours"] for s in stages)
    console.print(Panel.fit(f"[bold]Learning path for:[/bold] {goal}\nEstimated total time: {total:g} hours",
                            border_style="cyan"))
    step = 1
    for stage in stages:
        console.print(f"\n[bold underline]{stage['level']} stage[/] [dim]({stage['hours']:g} h)[/dim]")
        for r in stage["resources"]:
            console.print(f"  [cyan]Step {step}.[/] [bold]{r['title']}[/] [dim](ID {r['id']}, {r['format']}, "
                          f"{r['duration_hours']:g} h, {r['cost']})[/dim]")
            step += 1


def profile_panel(p, engine) -> None:
    lines = [
        f"Username: [bold]{p.username}[/]   Name: {p.name or '-'}",
        f"Skill level: {level(p.level)}",
        f"Interests: {', '.join(p.interests) or '-'}",
        f"Goal: {p.goals or '-'}",
        f"Preferred formats: {', '.join(p.preferred_formats) or 'Any'}",
        f"Max duration: {str(p.max_duration) + ' h' if p.max_duration else 'No limit'}   "
        f"Free only: {'Yes' if p.free_only else 'No'}",
        f"Rated: {len(p.ratings)}   Completed: {len(p.completed)}",
    ]
    console.print(Panel("\n".join(lines), title="My Profile", border_style="magenta"))
    if p.ratings or p.completed:
        table = Table(title="My learning history", header_style="bold magenta")
        table.add_column("ID", justify="right")
        table.add_column("Resource")
        table.add_column("Your rating")
        table.add_column("Completed")
        for rid in sorted(set(p.ratings) | set(p.completed)):
            r = engine.get_resource(rid)
            if not r:
                continue
            rating = p.ratings.get(rid)
            table.add_row(str(rid), r["title"], "★" * int(rating) if rating else "-",
                          "✔" if rid in p.completed else "")
        console.print(table)


def info(msg: str) -> None:
    console.print(f"[cyan]ℹ {msg}[/cyan]")


def success(msg: str) -> None:
    console.print(f"[green]✔ {msg}[/green]")


def error(msg: str) -> None:
    console.print(f"[red]✘ {msg}[/red]")
