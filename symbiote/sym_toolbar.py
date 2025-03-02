#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: sym_toolbar.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 15:56:34
# Modified: 2024-11-27 06:45:14
#!/usr/bin/env python3
#
# sym_toolbar.py

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from wcwidth import wcwidth
import psutil


def render_dashboard(settings=None, functions=None, max_lines=12, tail_lines=None):
    """
    Render a dashboard with system stats, function outputs, and a content viewer, returning ASCII content.

    Args:
        settings (dict): Key-value pairs to display in the dashboard.
        functions (dict): Key-function pairs; functions are executed, and their return values are displayed.
        max_lines (int): Maximum height of the dashboard.
        tail_lines (str | callable): File path to tail the last lines, or a callable to render content dynamically.

    Returns:
        str: The raw ASCII content of the rendered dashboard.
    """
    console = Console()
    col_pad = 0 
    

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
    dashboard_table = Table.grid(expand=False, padding=(0, 1))
    dashboard_table.add_column(justify="right", width=7)
    dashboard_table.add_column(justify="left")

    # Populate the dashboard with settings
    for key, value in settings.items():
        dashboard_table.add_row(f"{key}:", str(value))

    dashboard_panel = Panel(dashboard_table,
                            title="Dashboard",
                            title_align="left",
                            height=max_lines,
                            expand=True,
                            )

    # Handle content viewer (tail_lines)
    if callable(tail_lines):
        try:
            content = tail_lines()
            lines = content.splitlines()[-(max_lines - 6):]
        except Exception as e:
            lines = [f"Error: {e}"]
    elif isinstance(tail_lines, str):
        try:
            with open(tail_lines, "r") as file:
                lines = file.readlines()[-(max_lines):]
        except FileNotFoundError:
            lines = ["File not found."]
        except Exception as e:
            lines = [f"Error: {e}"]
    else:
        lines = ["Invalid content viewer input."]

    # Fill blank lines if needed
    while len(lines) < max_lines:
        lines.append("")

    display_text = "\n".join(lines)

    viewer_width = int(console.width * .75)
    log_panel = Panel(
        Text(display_text),
        title="Content Viewer",
        title_align="left",
        height=max_lines,
        expand=True,
        width=viewer_width,
    )

    char_head = "ￚￂￃￚￄￅￆￚￚￇￊￚￋￌￍￎￚￚￏￒￚￓￔￕￚￚￖￗￛￚￚￗￚￖￕￚￓￒￚￏￚￎￍￚￌￋￚￚￊￇￚￆￅￚￄￃￂￚ"
    char_foot = "￣"

    grid = Table.grid(expand=True)
    grid.add_column(justify="center")
    grid.add_row(Rule(characters=char_head, style=None))

    # Create the grid for the dashboard
    dashboard_grid = Table.grid(expand=True, padding=(0, 0))
    dashboard_grid.add_column(ratio=1)
    dashboard_grid.add_column(ratio=2)
    dashboard_grid.add_row(dashboard_panel, log_panel)

    grid.add_row(dashboard_grid)
    grid.add_row()
    #grid.add_row(Rule(characters=char_foot, style=None))
    
    # Use Group to organize the dashboard
    dashboard_group = Group(grid)

    # Capture and return the rendered dashboard as a string
    with console.capture() as capture:
        console.print(dashboard_group)
    ascii_render = capture.get()

    lines = ascii_render.splitlines()
    #lines = lines[:-1]
    ascii_render = "\n".join(lines)
    return ascii_render 

def main():
    # Example settings and function definitions
    example_settings = {"Model": "GPT-4", "Role": "Assistant", "Shell": "active"}
    example_functions = {
        "CPU": lambda: f"{psutil.cpu_percent()}%",
        "Memory": lambda: f"{psutil.virtual_memory().percent}%",
    }

    # Example file path or callable for content viewer
    example_tail_lines = "/var/log/system.log"  # Replace with a real file path
    # example_tail_lines = lambda: "Dynamic content from a function\nMore content..."

    # Render and print the dashboard
    dashboard = render_dashboard(
        settings=example_settings,
        functions=example_functions,
        max_lines=8,
        tail_lines=example_tail_lines,
    )
    print(dashboard.strip())

if __name__ == "__main__":
    main()
