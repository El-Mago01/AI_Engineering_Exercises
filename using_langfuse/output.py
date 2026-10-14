from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def nicer_print(user_input: str, available_dishes: list[str], meal_plan: str) -> None:
    """Render the request, menu, and meal plan as colorful terminal panels."""
    console.print(
        Panel(
            f"[italic]{user_input}[/italic]",
            title="🍽️  Generated request",
            border_style="cyan",
            title_align="left",
        )
    )
    console.print(
        Panel(
            f"[dim]{', '.join(available_dishes)}[/dim]",
            title="📋  Today's menu",
            border_style="yellow",
            title_align="left",
        )
    )
    console.print(
        Panel(
            Markdown(meal_plan),
            title="🤖  Meal plan",
            border_style="green",
            title_align="left",
        )
    )