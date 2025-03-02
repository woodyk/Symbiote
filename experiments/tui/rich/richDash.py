#!/usr/bin/env python3
#
# richDash.py

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text
import psutil
import time

# Initialize console and layout
console = Console()
PADDING = 0  # Padding for height and width
usable_height = console.height - PADDING
usable_width = console.width - PADDING

layout = Layout()
# Adjust the layout dynamically to fit within the terminal
# Total layout height is limited to usable_height
layout.split_column(
    Layout(name="header", size=3),  # Fixed header height
    Layout(name="main", size=usable_height - 6),  # Remaining space for main section
    Layout(name="footer", size=3),  # Fixed footer height
)

# Main section height distribution
main_height = usable_height - 6  # Header and footer take 6 lines total

layout["main"].split_row(
    Layout(name="column1", ratio=1),  # Equal width for 3 columns
    Layout(name="column2", ratio=1),
    Layout(name="column3", ratio=1),
)

# Column heights must not exceed main_height
layout["column1"].split_column(
    Layout(name="row1", size=main_height // 3),  # Divide height evenly across 3 rows
    Layout(name="row2", size=main_height // 3),
    Layout(name="row3", size=main_height // 3),
)

layout["column2"].split_column(
    Layout(name="large_row", size=main_height // 2),  # Large row takes 50% height
    Layout(name="small_row", size=main_height // 5),  # Small row takes 20% height
    Layout(name="remaining_row", size=main_height - (main_height // 2) - (main_height // 5)),
)

layout["small_row"].split_row(
    Layout(name="small_field1", ratio=1),  # Three equal fields
    Layout(name="small_field2", ratio=1),
    Layout(name="small_field3", ratio=1),
)

layout["column3"].split_column(
    Layout(name="field1", size=main_height // 3),
    Layout(name="field2", size=main_height // 3),
    Layout(name="field3", size=main_height // 3),
)

# --- Header ---
header_text = Text("⚡ SYSTEM DASHBOARD ⚡", justify="center", style="white")
layout["header"].update(Panel(header_text, border_style="white", title="Header", title_align="left"))

# --- Footer ---
footer_text = Text("Press [blue]Q[/blue] to quit | [yellow]ESC[/yellow] for help", justify="center")
layout["footer"].update(Panel(footer_text, border_style="white", title="Footer", title_align="left"))

# --- Column 1 Panels ---
def create_sample_table(title):
    table = Table(title=title)
    table.add_column("Metric", justify="right", style="cyan")
    table.add_column("Value", justify="left", style="yellow")
    table.add_row("CPU Load", f"{psutil.cpu_percent()}%")
    table.add_row("Memory", f"{psutil.virtual_memory().percent}%")
    table.add_row("Disk", f"{psutil.disk_usage('/').percent}%")
    return Panel(table, border_style="white", title=title, title_align="left")


layout["row1"].update(create_sample_table("System Metrics"))
layout["row2"].update(create_sample_table("Application Stats"))
layout["row3"].update(create_sample_table("Task Overview"))

# --- Column 2 Panels ---
def create_large_chart_panel():
    text = Text("Real-Time Metrics\n", style="magenta")
    text.append("Bing: [yellow]0.193s\n")
    text.append("Google: [green]0.516s\n")
    text.append("Yahoo: [blue]0.179s\n")
    return Panel(text, border_style="white", title="Large Chart", title_align="left")


layout["large_row"].update(create_large_chart_panel())

# Small Fields in Column 2
layout["small_field1"].update(Panel("Field 1 Content", border_style="white", title="Small Field 1", title_align="left"))
layout["small_field2"].update(Panel("Field 2 Content", border_style="white", title="Small Field 2", title_align="left"))
layout["small_field3"].update(Panel("Field 3 Content", border_style="white", title="Small Field 3", title_align="left"))

# Remaining Row in Column 2
def create_in_flight_messages_panel():
    table = Table(title="In-Flight Messages", border_style=None, box=None)
    table.add_column()
    table.add_column()
    table.add_row("Inbound", "1,000")
    table.add_row("Processing", "700")
    table.add_row("DLQ", "100")
    return Panel(table, border_style="white", title="In-Flight Messages", title_align="left")


layout["remaining_row"].update(create_in_flight_messages_panel())

# --- Column 3 Panels ---
def create_progress_panel(title, progress_percent):
    with Progress(
        TextColumn(f"{title}: {{task.description}}", justify="left"),
        BarColumn(),
        TextColumn("[green]{task.percentage:.1f}%"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]{title}", total=100)
        progress.update(task, completed=progress_percent)
    return Panel(progress, border_style="white", title=title, title_align="left")


layout["field1"].update(create_progress_panel("CPU Usage", psutil.cpu_percent()))
layout["field2"].update(create_progress_panel("Memory Usage", psutil.virtual_memory().percent))
layout["field3"].update(create_progress_panel("Disk Usage", psutil.disk_usage('/').percent))

# --- Render the layout ---
#console.clear()
with Live(layout, refresh_per_second=4) as live:
    while True:
        live.console.print(layout)
        time.sleep(2)

