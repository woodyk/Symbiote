#!/usr/bin/env python3
#
# richSlidingPanel.py

import time
from rich.console import Console


def sliding_panel(
    content: list,
    direction: str = "left",
    steps: int = 5,
    delay: float = 0.1,
    width: int = None,
):
    """
    Simulates a sliding panel animation in the terminal.

    Args:
        content (list): List of strings to display as the panel content.
        direction (str): Direction of the slide ('left' or 'top').
        steps (int): Number of steps for the sliding effect.
        delay (float): Delay between each step of the animation.
        width (int): Width of the terminal for centering. Defaults to terminal width.
    """
    console = Console()
    terminal_width = console.size.width if width is None else width

    if direction == "left":
        for step in range(terminal_width, -1, -steps):
            console.clear()
            for line in content:
                console.print(" " * step + line)
            time.sleep(delay)
    elif direction == "top":
        blank_lines = len(content)
        for step in range(blank_lines + steps, -1, -1):
            console.clear()
            visible_content = content[: len(content) - max(step, 0)]
            for line in visible_content:
                console.print(line.center(terminal_width))
            time.sleep(delay)

    console.print("\n".join(content).center(terminal_width))


# Example Usage
if __name__ == "__main__":
    panel_content = [
        "╔═══════════════════════════╗",
        "║     Sliding Panel Demo    ║",
        "║      Rich is Awesome!     ║",
        "╚═══════════════════════════╝",
    ]

    # Slide from the left
    sliding_panel(panel_content, direction="left", steps=3, delay=0.05)

    # Slide from the top
    sliding_panel(panel_content, direction="top", steps=1, delay=0.1)

