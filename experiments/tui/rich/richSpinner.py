#!/usr/bin/env python3
#
# richSpinner.py

import time
from rich.console import Console
from rich.spinner import Spinner
from rich.table import Table

# Initialize Console
console = Console()

spinner_names = [
    "dots", "dots2", "dots3", "dots4", "dots5", "dots6", "dots7", "dots8", "dots9",
    "dots10", "dots11", "dots12", "line", "line2", "pipe", "simpleDots", "simpleDotsScrolling",
    "star", "star2", "flip", "hamburger", "growVertical", "growHorizontal", "balloon",
    "balloon2", "noise", "bounce", "boxBounce", "boxBounce2", "triangle", "arc", "circle",
    "squareCorners", "circleQuarters", "circleHalves", "squish", "toggle", "toggle2", "toggle3",
    "toggle4", "toggle5", "toggle6", "toggle7", "toggle8", "toggle9", "toggle10", "toggle11",
    "toggle12", "toggle13", "arrow", "arrow2", "arrow3", "bouncingBar", "bouncingBall",
    "smiley", "monkey", "hearts", "clock", "earth", "moon", "runner", "pong", "shark", "dqpb",
    "weather", "christmas", "grenade", "point", "layer", "betaWave",
]

# Create a table
spinner_table = Table(title="Static Spinner Samples", header_style="bold cyan")
spinner_table.add_column("Spinner Name", justify="left", style="bold yellow")
spinner_table.add_column("Static Sample", justify="center", style="bold green")

# Add spinner samples to the table
for spinner_name in spinner_names:
    spinner = Spinner(spinner_name, text="Sample")
    static_frame = spinner.render(time=0)  # Render the first frame of the spinner
    spinner_table.add_row(spinner_name, static_frame)

console.print(spinner_table)

# Example 1: Basic Spinner
console.rule("Example 1: Basic Spinner")
spinner = Spinner("dots", text="Loading...")
with console.status(spinner):
    time.sleep(0.5)

# Example 2: Spinner with Custom Text and Style
console.rule("Example 2: Custom Text and Style")
custom_spinner = Spinner("dots", text="[bold green]Custom Loading[/bold green]", style="bold cyan")
with console.status(custom_spinner):
    time.sleep(0.5)

# Example 3: Adjust Spinner Speed
console.rule("Example 3: Spinner with Adjusted Speed")
fast_spinner = Spinner("bouncingBall", text="Fast Loading...", speed=2.0)
with console.status(fast_spinner):
    time.sleep(0.5)

slow_spinner = Spinner("bouncingBall", text="Slow Loading...", speed=0.5)
with console.status(slow_spinner):
    time.sleep(0.5)

# Example 4: Dynamic Updates to Spinner
console.rule("Example 4: Dynamic Updates to Spinner")
dynamic_spinner = Spinner("dots", text="Initial Loading...")
with console.status(dynamic_spinner):
    time.sleep(1)
    dynamic_spinner.update(text="[bold yellow]Updated Text[/bold yellow]", style="bold red", speed=1.5)
    time.sleep(0.5)

# Example 5: Showcase of All Built-In Spinners
console.rule("Example 5: All Built-In Spinners")

for spinner_name in spinner_names:
    console.print(f"Spinner Name: [bold cyan]{spinner_name}[/bold cyan]")
    spinner = Spinner(spinner_name, text=f"Spinner: {spinner_name}")
    with console.status(spinner):
        time.sleep(0.5)

# Example 6: Custom Renderable for Spinner Text
from rich.text import Text
console.rule("Example 6: Custom Renderable Text")
custom_text = Text("Custom Spinner Text", style="bold magenta underline")
custom_text_spinner = Spinner("dots", text=custom_text)
with console.status(custom_text_spinner):
    time.sleep(0.5)

# Example 7: Using Spinner Without Console Status
console.rule("Example 7: Spinner Without Console Status")
standalone_spinner = Spinner("dots", text="Standalone Spinner")
console.print(standalone_spinner.render(time.time()))

# Summary of Features
console.rule("Summary of Spinner Features")
summary = """
1. Basic Spinner: Simple spinner with default settings.
2. Custom Text and Style: Customize spinner text and style.
3. Adjusted Speed: Control spinner speed for fast or slow animations.
4. Dynamic Updates: Change spinner attributes dynamically.
5. Built-In Spinners: Display all available spinner types.
6. Custom Text Renderable: Use `Text` instances for advanced spinner text.
7. Standalone Spinner: Render spinner without `console.status`.
"""
console.print(summary)

