"""
Console helpers shared by the demo scripts.

step_banner() prints a framed header between pipeline steps and, in live
demo mode, waits for Enter so the presenter can narrate one step at a time.
"""

import shutil

WIDTH = min(shutil.get_terminal_size(fallback=(72, 24)).columns, 72)

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"


def step_banner(label: str, pause: bool = True) -> None:
    """Print a separator block for the next step, then optionally pause."""
    rule = "━" * WIDTH
    print(f"\n{CYAN}{rule}{RESET}")
    print(f"{BOLD}  ▶ {label}{RESET}")
    print(f"{CYAN}{rule}{RESET}")
    if pause:
        try:
            input(f"{DIM}  Press Enter to continue...{RESET}")
        except EOFError:
            # Non-interactive run (piped input) - carry on without pausing.
            pass
        print()
