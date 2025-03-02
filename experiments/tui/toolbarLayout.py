#!/usr/bin/env python3
#
# toolbar.py

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
import psutil


def render_dashboard(settings, shell_mode, log_file_path, height=12):
    """
    Render a dashboard with system stats and log tailing, returning the ASCII content.

    Args:
        settings (dict): A dictionary containing 'model' and 'role'.
        shell_mode (str): The shell mode.
        log_file_path (str): Path to the log file to tail.
        height (int): The height of the rendered dashboard in console lines.

    Returns:
        str: The raw ASCII content of the rendered dashboard.
    """
    console = Console()

    # Column 1: Key-Value Pairs for Settings and System Info
    column1_table = Table.grid(expand=True, padding=(0, 1))
    column1_table.add_column(justify="left", width=12)
    column1_table.add_column(justify="left", width=13)

    # Add settings rows
    column1_table.add_row("Model:", settings.get("model", "Unknown"))
    column1_table.add_row("Role:", settings.get("role", "Unknown"))
    column1_table.add_row("Shell Mode:", shell_mode)
    column1_table.add_row("", "")  # Empty row for spacing

    # Add system stats
    column1_table.add_row("CPU:", f"{psutil.cpu_percent()}%")
    column1_table.add_row("Memory:", f"{psutil.virtual_memory().percent}%")
    disk_usage = psutil.disk_usage("/")
    column1_table.add_row("Disk:", f"{disk_usage.percent}%")
    net_io = psutil.net_io_counters()
    column1_table.add_row("Net In:", f"{net_io.bytes_recv / (1024**2):.2f} MB")
    column1_table.add_row("Net Out:", f"{net_io.bytes_sent / (1024**2):.2f} MB")

    column1_panel = Panel(column1_table, title="Dashboard", title_align="left")

    # Column 2: Tail of Log File
    try:
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()[-(height - 6) :]  # Adjust tailing to fit height
    except FileNotFoundError:
        lines = ["Log file not found."]
    except Exception as e:
        lines = [f"Error reading log file: {e}"]

    log_panel = Panel("\n".join(lines), title="Log Tail", title_align="left")

    # Create layout
    layout = Layout()
    layout.split_row(
        Layout(column1_panel, name="left", ratio=1),
        Layout(log_panel, name="right", ratio=3),
    )

    # Capture and return the rendered dashboard as a string
    with console.capture() as capture:
        console.print(layout)
    return capture.get()


# Example Usage
if __name__ == "__main__":
    example_settings = {"model": "GPT-4", "role": "Assistant"}
    example_shell_mode = "Interactive"
    example_log_file = "/path/to/your/logfile.log"  # Replace with a real log file path
    dashboard = render_dashboard(example_settings, example_shell_mode, example_log_file, height=12)

    # Print the dashboard
    print(dashboard)

