#!/usr/bin/env python3
#
# richGridDash.py

from rich.console import Console, Group
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich.text import Text
import psutil
import time

# Initialize console
console = Console()
console_width = console.width
console_height = console.height

# Calculate padding (approx 5 lines for breathing space)
available_height = console_height - 5

# Split heights based on the layout structure
column1_row_height = available_height // 3
column2_row1_height = available_height // 2
column2_row2_height = available_height // 5
column2_row3_height = available_height - (column2_row1_height + column2_row2_height)


# Helper functions for panels
def search_results_panel():
    text = Text("Search Results / File Viewer\n", style="yellow")
    text.append("[INFO] Task completed successfully.\n")
    text.append("[WARN] High memory usage detected.\n")
    text.append("[ERROR] Unable to load configuration.\n")
    return Panel(
        text,
        title="Search Results",
        border_style="white",
        title_align="left",
        height=column1_row_height,
    )


def event_log_panel():
    text = Text("Event Log\n", style="cyan")
    text.append("[12:34] User logged in.\n")
    text.append("[12:36] File uploaded.\n")
    text.append("[12:40] System error detected.\n")
    return Panel(
        text,
        title="Event Log",
        border_style="white",
        title_align="left",
        height=column1_row_height,
    )


def resource_usage_panel():
    table = Table(show_header=True, header_style="white")
    table.add_column("Resource", justify="left")
    table.add_column("Usage", justify="right")
    table.add_row("CPU", f"{psutil.cpu_percent()}%")
    table.add_row("Memory", f"{psutil.virtual_memory().percent}%")
    table.add_row("Disk", f"{psutil.disk_usage('/').percent}%")
    return Panel(
        table,
        title="Resource Usage",
        border_style="white",
        title_align="left",
        height=column1_row_height,
    )


def response_time_panel():
    text = Text("Search Engine Response Times\n", style="magenta")
    text.append("Google: 0.512s\n", style="green")
    text.append("Bing: 0.184s\n", style="yellow")
    text.append("Yahoo: 0.205s\n", style="blue")
    return Panel(
        text,
        title="Response Times",
        border_style="white",
        title_align="left",
        height=column2_row1_height,
    )


def small_panel(title, content, height):
    return Panel(
        Text(content, style="cyan"),
        title=title,
        border_style="white",
        title_align="left",
        height=height,
    )


def process_monitor_panel():
    table = Table(show_header=True, header_style="white")
    table.add_column("PID", justify="right")
    table.add_column("Process", justify="left")
    table.add_column("CPU%", justify="right")
    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
        table.add_row(str(proc.info["pid"]), proc.info["name"], str(proc.info["cpu_percent"]))
        if len(table.rows) >= 5:
            break
    return Panel(
        table,
        title="Process Monitor",
        border_style="white",
        title_align="left",
        height=column2_row3_height,
    )


def clock_panel():
    clock_text = Text(time.strftime("%H:%M:%S"), style="bold cyan")
    return Panel(
        clock_text,
        title="Clock",
        border_style="white",
        title_align="left",
        height=column2_row2_height,
    )


# Create the layout
column1 = Columns(
    [
        search_results_panel(),
        event_log_panel(),
        resource_usage_panel(),
    ],
    align="left",
)

column2_row1 = response_time_panel()

column2_row2 = Columns(
    [
        small_panel("Year Progress", "80%", column2_row2_height),
        small_panel("Month Progress", "45%", column2_row2_height),
        clock_panel(),
    ]
)

column2_row3 = process_monitor_panel()

column2 = Group(column2_row1, column2_row2, column2_row3)

# Display the final dashboard
dashboard = Columns([column1, column2], align="center", expand=True)
console.clear()
console.print(dashboard)

