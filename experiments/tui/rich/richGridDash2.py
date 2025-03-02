#!/usr/bin/env python3
#
# richGridDash2.py

from rich.console import Console, Group
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich.text import Text
import psutil
import time

# Initialize console and calculate height limits
console = Console()
padding = 4
max_height = console.height - padding


# Helper Functions for Panels
def search_results_panel(height):
    text = Text("Search Results / File Viewer\n", style="yellow")
    text.append("[INFO] Task completed successfully.\n")
    text.append("[WARN] High memory usage detected.\n")
    text.append("[ERROR] Unable to load configuration.\n")
    return Panel(text, title="Search Results", border_style="white", title_align="left", height=height)


def event_log_panel(height):
    text = Text("Event Log\n", style="cyan")
    text.append("[12:34] User logged in.\n")
    text.append("[12:36] File uploaded.\n")
    text.append("[12:40] System error detected.\n")
    return Panel(text, title="Event Log", border_style="white", title_align="left", height=height)


def resource_usage_panel(height):
    table = Table(show_header=True, header_style="white")
    table.add_column("Resource", justify="left")
    table.add_column("Usage", justify="right")
    table.add_row("CPU", f"{psutil.cpu_percent()}%")
    table.add_row("Memory", f"{psutil.virtual_memory().percent}%")
    table.add_row("Disk", f"{psutil.disk_usage('/').percent}%")
    return Panel(table, title="Resource Usage", border_style="white", title_align="left", height=height)


def response_time_panel(height):
    text = Text("Search Engine Response Times\n", style="magenta")
    text.append("Google: 0.512s\n", style="green")
    text.append("Bing: 0.184s\n", style="yellow")
    text.append("Yahoo: 0.205s\n", style="blue")
    return Panel(text, title="Response Times", border_style="white", title_align="left", height=height)


def process_monitor_panel(height):
    table = Table(show_header=True, header_style="white")
    table.add_column("PID", justify="right")
    table.add_column("Process", justify="left")
    table.add_column("CPU%", justify="right")
    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
        table.add_row(str(proc.info["pid"]), proc.info["name"], str(proc.info["cpu_percent"]))
        if len(table.rows) >= 5:
            break
    return Panel(table, title="Process Monitor", border_style="white", title_align="left", height=height)


def clock_panel(height):
    clock_text = Text(time.strftime("%H:%M:%S"), style="bold cyan")
    return Panel(clock_text, title="Clock", border_style="white", title_align="left", height=height)


# Construct Dashboard Sections
section1 = Group(
    Columns([search_results_panel(max_height // 3), event_log_panel(max_height // 3), resource_usage_panel(max_height // 3)], align="left")
)

section2 = Group(
    response_time_panel(max_height // 2),
    Columns(
        [
            Panel(Text("Year Progress: 80%", style="cyan"), title="Year Progress", border_style="white", title_align="left", height=max_height // 5),
            Panel(Text("Month Progress: 45%", style="cyan"), title="Month Progress", border_style="white", title_align="left", height=max_height // 5),
            clock_panel(max_height // 5),
        ]
    ),
    process_monitor_panel(max_height // 3),
)

# Encapsulate sections into main panels
dashboard_part1 = Panel(section1, title="Dashboard - Part 1", border_style="white", title_align="left", height=max_height)
dashboard_part2 = Panel(section2, title="Dashboard - Part 2", border_style="white", title_align="left", height=max_height)

# Render panels one at a time
console.clear()
console.print(dashboard_part1)
console.print("\n" + "=" * console.width + "\n")  # Separator between parts
console.print(dashboard_part2)

