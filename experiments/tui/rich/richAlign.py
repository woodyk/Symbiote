#!/usr/bin/env python3
#
# richAlign.py

from rich.console import Console
from rich.align import Align, VerticalCenter
from rich.panel import Panel
from rich.text import Text

# Initialize Console
console = Console()

# Example renderable: A panel with some text
renderable = Panel("This is an example of text alignment using rich.align.", title="Rich Align")

# Align to the Left
console.rule("Align Left Example")
align_left = Align.left(renderable, width=50, height=10, style="magenta", vertical="top")
console.print(align_left)

# Align to the Right
console.rule("Align Right Example")
align_right = Align.right(renderable, width=50, height=10, style="cyan", vertical="top")
console.print(align_right)

# Align to the Center
console.rule("Align Center Example")
align_center = Align.center(renderable, width=50, height=10, style="green", vertical="middle")
console.print(align_center)

# Vertical Alignment Example: Middle
console.rule("Vertical Middle Alignment Example")
vertical_middle = Align(
    renderable, align="center", width=50, height=10, style="yellow", vertical="middle"
)
console.print(vertical_middle)

# Vertical Alignment Example: Top
console.rule("Vertical Top Alignment Example")
vertical_top = Align(
    renderable, align="center", width=50, height=10, style="blue", vertical="top"
)
console.print(vertical_top)

# Vertical Alignment Example: Bottom
console.rule("Vertical Bottom Alignment Example")
vertical_bottom = Align(
    renderable, align="center", width=50, height=10, style="red", vertical="bottom"
)
console.print(vertical_bottom)

# Using Align without Style
console.rule("Align Without Style Example")
align_no_style = Align(renderable, align="center", width=50, height=10)
console.print(align_no_style)

# Using VerticalCenter (Deprecated)
console.rule("VerticalCenter Example (Deprecated)")
vertical_center = VerticalCenter(renderable, style="white")
console.print(vertical_center)

# Summary of alignments
console.rule("Summary of Alignments")
summary_text = Text.from_markup(
    "[bold green]1. Left Alignment:[/] Text is aligned to the left within the width.\n"
    "[bold cyan]2. Right Alignment:[/] Text is aligned to the right within the width.\n"
    "[bold yellow]3. Center Alignment:[/] Text is centered horizontally and vertically.\n"
    "[bold blue]4. Vertical Alignments:[/] Top, Middle, Bottom vertical positioning.\n"
    "[bold white]5. VerticalCenter (Deprecated):[/] Vertically centers text (use Align with vertical='middle')."
)
console.print(Panel(summary_text, title="Summary"))

