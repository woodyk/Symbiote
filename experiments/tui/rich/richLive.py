#!/usr/bin/env python3
#
# richLive.py

import time
import random
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

console = Console()

# Helper function to create a dynamic table
def generate_dynamic_table() -> Table:
    table = Table()
    table.add_column("ID")
    table.add_column("Value")
    table.add_column("Status")

    for row in range(random.randint(3, 6)):
        value = random.uniform(0, 100)
        status = "[green]SUCCESS" if value > 50 else "[red]ERROR"
        table.add_row(f"{row}", f"{value:.2f}", status)
    return table


### Example 1: Simple Live Updating Table ###
console.rule("[bold cyan]Example 1: Simple Live Updating Table[/bold cyan]")

table = Table(title="Live Table")
table.add_column("Row ID", style="cyan")
table.add_column("Description", style="magenta")
table.add_column("Level", style="green")

with Live(table, refresh_per_second=4, transient=True):
    for i in range(10):
        time.sleep(0.5)
        table.add_row(f"{i}", f"description {i}", "[red]ERROR" if i % 2 else "[green]OK")

time.sleep(2)  # Pause before the next example
console.clear()


### Example 2: Dynamic Renderable Updates ###
console.rule("[bold cyan]Example 2: Dynamic Renderable Updates[/bold cyan]")

with Live(generate_dynamic_table(), refresh_per_second=4) as live:
    for _ in range(15):
        time.sleep(0.5)
        live.update(generate_dynamic_table())  # Update with new renderable

time.sleep(2)
console.clear()


### Example 3: Alternate Screen Mode ###
console.rule("[bold cyan]Example 3: Alternate Screen Mode[/bold cyan]")

with Live(Panel("This is an alternate screen demo. Watch as it clears on exit."), screen=True, refresh_per_second=2):
    time.sleep(5)

time.sleep(2)
console.clear()


### Example 4: Transient Display ###
console.rule("[bold cyan]Example 4: Transient Display[/bold cyan]")

with Live(Panel("This display will disappear after execution."), transient=True):
    time.sleep(3)

time.sleep(2)
console.clear()


### Example 5: Logging Above the Live Display ###
console.rule("[bold cyan]Example 5: Logging Above the Live Display[/bold cyan]")

table = Table(title="Log and Table Example")
table.add_column("Row ID")
table.add_column("Description")
table.add_column("Level")

with Live(table, refresh_per_second=4) as live:
    for i in range(8):
        live.console.log(f"Processing row {i}...")  # Log above live display
        time.sleep(0.5)
        table.add_row(f"{i}", f"description {i}", "[green]OK" if i % 2 == 0 else "[red]ERROR")

time.sleep(2)
console.clear()


### Example 6: Handling Vertical Overflow ###
console.rule("[bold cyan]Example 6: Handling Vertical Overflow[/bold cyan]")

large_table = Table(title="Overflow Example")
large_table.add_column("Index", justify="right")
large_table.add_column("Data")

for i in range(50):  # Add many rows to simulate overflow
    large_table.add_row(f"{i}", f"Data {i}")

with Live(large_table, vertical_overflow="ellipsis", refresh_per_second=2):
    time.sleep(5)

time.sleep(2)
console.clear()


### Example 7: Interactive Progress with Nested Layout ###
console.rule("[bold cyan]Example 7: Interactive Progress[/bold cyan]")

from rich.layout import Layout
from rich.progress import Progress

layout = Layout(name="root")
layout.split_column(
    Layout(name="header", size=3),
    Layout(name="body"),
    Layout(name="footer", size=3),
)
layout["header"].update(Panel("[cyan]Live Progress Example[/cyan]"))
layout["footer"].update(Panel("[green]Footer Section[/green]"))

progress = Progress()
task = progress.add_task("Processing...", total=100)
layout["body"].update(progress)

with Live(layout, refresh_per_second=10):
    for _ in range(100):
        time.sleep(0.1)
        progress.update(task, advance=1)

console.clear()
console.print("[bold green]All examples completed![/bold green]")

