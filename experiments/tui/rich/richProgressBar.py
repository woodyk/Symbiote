#!/usr/bin/env python3
#
# richProgressBar.py

import time
from rich.console import Console
from rich.progress_bar import ProgressBar
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn, track
from rich.panel import Panel

# Initialize Console
console = Console()

# Example 1: Basic Progress Bar
console.rule("Basic Progress Bar")
basic_progress_bar = ProgressBar(total=100, completed=50)
console.print(basic_progress_bar)

# Example 2: Pulsing Progress Bar
console.rule("Pulsing Progress Bar")
pulsing_bar = ProgressBar(total=None, pulse=True, pulse_style="magenta")
console.print(pulsing_bar)

# Example 3: Styled Progress Bar
console.rule("Styled Progress Bar")
styled_bar = ProgressBar(
    total=100,
    completed=30,
    style="on blue",
    complete_style="bold green",
    finished_style="bold red",
)
console.print(styled_bar)

# Example 4: Progress Bar with Custom Width
console.rule("Custom Width Progress Bar")
width_bar = ProgressBar(total=100, completed=40, width=20)
console.print(width_bar)

# Example 5: Updating Progress Bar
console.rule("Updating Progress Bar")
updating_bar = ProgressBar(total=100, completed=0)
for i in range(101):
    updating_bar.update(completed=i)
    console.print(updating_bar, end="\r")
    time.sleep(0.05)
console.print("\n")

# Example 6: Integration with `rich.progress`
console.rule("Integration with rich.progress")
with Progress() as progress:
    task = progress.add_task("[green]Processing...", total=100)
    while not progress.finished:
        progress.update(task, advance=1)
        time.sleep(0.05)

# Example 7: Using `track` for Simplified Progress
console.rule("Using track() for Simplified Progress")
for _ in track(range(50), description="[cyan]Simplified Progress..."):
    time.sleep(0.1)

# Example 8: Transient Progress (disappears after completion)
console.rule("Transient Progress Example")
with Progress(transient=True) as progress:
    task = progress.add_task("[blue]Temporary Task...", total=100)
    for _ in range(100):
        progress.update(task, advance=1)
        time.sleep(0.05)

# Example 9: Indeterminate Progress Bar
console.rule("Indeterminate Progress Example")
indeterminate_bar = ProgressBar(total=None, pulse=True, animation_time=0.5, pulse_style="dim yellow")
console.print(indeterminate_bar)

# Example 10: Multiple Progress Bars
console.rule("Multiple Progress Bars")
with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
) as progress:
    task1 = progress.add_task("[red]Downloading...", total=200)
    task2 = progress.add_task("[green]Processing...", total=100)
    while not progress.finished:
        progress.update(task1, advance=2)
        progress.update(task2, advance=1)
        time.sleep(0.05)

# Example 11: Custom Progress Bar Columns
console.rule("Custom Progress Bar Columns")
from rich.table import Column

text_column = TextColumn("{task.description}", table_column=Column(ratio=1))
bar_column = BarColumn(bar_width=None, table_column=Column(ratio=2))
with Progress(text_column, bar_column, expand=True) as progress:
    for _ in progress.track(range(100), description="Custom Columns"):
        time.sleep(0.1)

# Example 12: Reading a File with Progress
'''
console.rule("Reading File with Progress")
import json
from rich.progress import open as rich_open

with rich_open("example.json", "rb") as file:  # Replace with a valid JSON file
    data = json.load(file)

console.print(Panel("JSON File Loaded Successfully!", title="File Read"))
'''

# Example 13: Multiple Progress Instances
console.rule("Multiple Progress Instances Example")
from rich.live import Live

progress1 = Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn())
progress2 = Progress(*Progress.get_default_columns(), TimeRemainingColumn())
live = Live(console=console)

with live:
    with progress1, progress2:
        task1 = progress1.add_task("First Task", total=50)
        task2 = progress2.add_task("Second Task", total=100)
        while not (progress1.finished and progress2.finished):
            progress1.update(task1, advance=1)
            progress2.update(task2, advance=2)
            time.sleep(0.05)

# Summary of Features
console.rule("Summary of ProgressBar Features")
summary = """
1. Basic Progress Bar: Simple bar with total and completed.
2. Pulsing Bar: Indeterminate progress with animation.
3. Styled Bar: Custom background, complete, and finished styles.
4. Custom Width: Define a fixed width for the bar.
5. Updating: Dynamically update progress values.
6. Integration with rich.progress: Use with tasks and bars.
7. track(): Simplify progress with iterable tracking.
8. Transient Progress: Temporary progress that disappears.
9. Indeterminate Progress: No total, just a visual pulse.
10. Multiple Progress Bars: Concurrent tasks in one view.
11. Custom Columns: Add or customize progress columns.
12. File Reading: Display progress while reading files.
13. Multiple Progress Instances: Handle multiple displays using `Live`.
"""
console.print(Panel(summary, title="Progress Bar Features", style="bold cyan"))

