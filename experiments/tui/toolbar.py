#!/usr/bin/env python3
#
# toolbar.py

from rich.console import Console, Group
from rich.text import Text
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table
import psutil


def render_dashboard(settings, shell_mode, log_file_path, max_lines=8):
    """
    Render a dashboard with system stats and log tailing, returning the ASCII content.

    Args:
        settings (dict): A dictionary containing 'model' and 'role'.
        shell_mode (str): The shell mode.
        log_file_path (str): Path to the log file to tail.
        max_lines (int): The maximum height of the dashboard.

    Returns:
        str: The raw ASCII content of the rendered dashboard.
    """
    console = Console()

    # Column 1: Key-Value Pairs for Settings and System Info
    dashboard_table = Table.grid(expand=True, padding=(0, 0))
    dashboard_table.add_column(justify="left", width=None)
    dashboard_table.add_column(justify="left", width=None)

    # Add settings rows
    for key, value in settings.items():
        dashboard_table.add_row(key, value)

    # Add system stats
    '''
    dashboard_table.add_row("CPU:", f"{psutil.cpu_percent()}%")
    dashboard_table.add_row("Memory:", f"{psutil.virtual_memory().percent}%")
    disk_usage = psutil.disk_usage("/")
    dashboard_table.add_row("Disk:", f"{disk_usage.percent}%")
    net_io = psutil.net_io_counters()
    dashboard_table.add_row("Net In:", f"{net_io.bytes_recv / (1024**2):.2f} MB")
    dashboard_table.add_row("Net Out:", f"{net_io.bytes_sent / (1024**2):.2f} MB")
    '''

    dashboard_panel = Panel(dashboard_table, title="Dashboard", title_align="left", height=max_lines)

    # Column 2: Tail of Log File
    max_log_lines = max(1, max_lines - 6)  # Adjust log lines to fit max_lines
    try:
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()[-max_log_lines:]  # Tail last n lines
    except FileNotFoundError:
        lines = ["Log file not found."]
    except Exception as e:
        lines = [f"Error reading log file: {e}"]

    while len(lines) < max_lines:
        lines += "\n"

    log_panel = Panel(Syntax("\n".join(lines), "python"), title="Log Tail", title_align="left", expand=True, height=max_lines)

    # Create the grid for the dashboard
    dashboard_grid = Table.grid(expand=True, padding=(0, 0))
    dashboard_grid.add_column(ratio=1)
    dashboard_grid.add_column(ratio=3)

    dashboard_grid.add_row(dashboard_panel, log_panel)

    # Use Group to organize the dashboard and constrain the height
    dashboard_group = Group(dashboard_grid)

    # Capture and return the rendered dashboard as a string
    with console.capture() as capture:
        console.print(dashboard_group)
    return capture.get()


# Example Usage
if __name__ == "__main__":
    example_settings = {"model": "GPT-4", "role": "Assistant", "mode": "shell"}
    example_shell_mode = "Interactive"
    example_log_file = "/var/log/system.log" # Replace with a real log file path
    dashboard = render_dashboard(example_settings, example_shell_mode, example_log_file)

    # Print the rendered dashboard
    print(dashboard.strip())

