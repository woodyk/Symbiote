#!/usr/bin/env python3
#
# sym_widgets.py

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
import subprocess
import psutil


class DashboardWidget:
    """
    A widget for rendering a dashboard with system stats, functions, and content viewer.
    """
    def __init__(self):
        self.console = Console()

    def render_dashboard(self, settings=None, functions=None, max_lines=12, tail_lines=None, shell=None):
        """
        Render a dashboard with settings, function outputs, and a content viewer.

        Args:
            settings (dict): Key-value pairs to display in the dashboard.
            functions (dict): Key-function pairs; functions are executed, and their return values are displayed.
            max_lines (int): Maximum height of the dashboard.
            tail_lines (str | callable): File path to tail the last lines, or a callable to render content dynamically.
            shell (str): Shell command to execute and display output in the Content Viewer.

        Returns:
            str: The raw ASCII content of the rendered dashboard.
        """
        # Initialize settings and functions
        settings = settings or {}
        functions = functions or {}

        # Execute functions and add their outputs to settings
        for key, func in functions.items():
            try:
                settings[key] = func() if callable(func) else "Invalid Function"
            except Exception as e:
                settings[key] = f"Error: {e}"

        # Create the dashboard panel
        dashboard_table = Table.grid(expand=True, padding=(0, 1))
        dashboard_table.add_column(justify="left", width=12)
        dashboard_table.add_column(justify="left", width=13)

        # Populate the dashboard with settings
        for key, value in settings.items():
            dashboard_table.add_row(f"{key}:", str(value))

        dashboard_panel = Panel(dashboard_table, title="Dashboard", title_align="left", height=max_lines)

        # Handle content viewer (shell commands, tail_lines, or fallback)
        if shell:
            try:
                result = subprocess.check_output(shell, shell=True, text=True, stderr=subprocess.STDOUT)
                lines = result.splitlines()[-(max_lines - 6):]
            except subprocess.CalledProcessError as e:
                lines = [f"Shell command failed: {e}"]
        elif callable(tail_lines):
            try:
                content = tail_lines()
                lines = content.splitlines()[-(max_lines - 6):]
            except Exception as e:
                lines = [f"Error: {e}"]
        elif isinstance(tail_lines, str):
            try:
                with open(tail_lines, "r") as file:
                    lines = file.readlines()[-(max_lines - 6):]
            except FileNotFoundError:
                lines = ["File not found."]
            except Exception as e:
                lines = [f"Error: {e}"]
        else:
            lines = ["No content provided."]

        # Fill blank lines if needed
        while len(lines) < max_lines:
            lines.append("")

        content_panel = Panel(
            Syntax("\n".join(lines), "plaintext"),
            title="Content Viewer",
            title_align="left",
            height=max_lines,
            expand=True,
        )

        # Create the grid for the dashboard
        dashboard_grid = Table.grid(expand=True, padding=(0, 0))
        dashboard_grid.add_column(ratio=1)
        dashboard_grid.add_column(ratio=3)
        dashboard_grid.add_row(dashboard_panel, content_panel)

        # Use Group to organize the dashboard
        dashboard_group = Group(dashboard_grid)

        # Capture and return the rendered dashboard as a string
        with self.console.capture() as capture:
            self.console.print(dashboard_group)
        return capture.get()

def main():
    # from symbiote.sym_widets import DashboardWidget
    # Example functions
    def get_cpu():
        return f"{psutil.cpu_percent()}%"

    def get_memory():
        return f"{psutil.virtual_memory().percent}%"


    if __name__ == "__main__":
        # Initialize widget
        dashboard = DashboardWidget()

        # Define settings and functions
        example_settings = {"Model": "GPT-4", "Role": "Assistant"}
        example_functions = {
            "CPU": get_cpu,
            "Memory": get_memory,
        }

        # Example file path or shell command for content viewer
        example_tail_lines = "/var/log/system.log"  # Replace with a real file path
        example_shell_command = "df -h"

        # Render the dashboard with shell command
        result = dashboard.render_dashboard(
            settings=example_settings,
            functions=example_functions,
            max_lines=12,
            tail_lines=example_tail_lines,
            shell=example_shell_command,
        )

        # Print the dashboard output
        print(result.strip())



if __name__ == "__main__":
    main()
