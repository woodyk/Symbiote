#!/usr/bin/env python3
#
# richGridFullscreenFillDashPanel.py

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn
from rich.align import Align

# Initialize Console
console = Console()

# Terminal dimensions
terminal_size = console.size
total_width = terminal_size.width
total_height = terminal_size.height - 2  # Subtract 2 for the terminal prompt area

# Number of rows and columns for the grid
cols = 3
row_height = 3  # Approximate height per row (header, content, padding)
rows = (total_height - 4) // row_height  # -4 for panel borders and rule

# Create Widgets
def create_widget(title, content, style="white on black"):
    """Creates a styled widget with a title and content."""
    panel = Panel(
        Text(content, justify="center"),
        title=title,
        title_align="left",
        style=style,
        padding=(1, 2),
    )
    return panel

# Example widgets
widgets = [
    create_widget("Server Status", "[bold green]All Systems Operational[/bold green]"),
    create_widget("CPU Usage", "[bold cyan]32%[/bold cyan]"),
    create_widget("Memory Usage", "[bold cyan]58%[/bold cyan]"),
    create_widget("Disk Space", "[bold yellow]3.2 TB Free of 5 TB[/bold yellow]"),
    create_widget("Notifications", "⚠️ [bold red]3 New Alerts[/bold red]"),
    create_widget("Tasks", "📋 12 Pending"),
    create_widget("Active Users", "[bold magenta]150[/bold magenta] Online"),
    create_widget("Network Traffic", "⬆️ [bold cyan]120 Mbps[/bold cyan]\n⬇️ [bold cyan]90 Mbps[/bold cyan]"),
]

# Add progress bar widget
progress_widget = Progress(
    SpinnerColumn(),
    BarColumn(),
    TaskProgressColumn(),
    expand=True,
    transient=True,
)
progress_task = progress_widget.add_task("Processing Data", total=100, completed=64)
widgets.append(Align(progress_widget, align="center"))

# Construct Full-Screen Dashboard
dashboard = Table.grid(expand=True)

# Add columns dynamically
for _ in range(cols):
    dashboard.add_column(ratio=1)

# Populate rows with widgets
widget_index = 0
for _ in range(rows):
    row_content = []
    for _ in range(cols):
        if widget_index < len(widgets):
            row_content.append(widgets[widget_index])
            widget_index += 1
        else:
            row_content.append("")  # Empty cell if no more widgets
    dashboard.add_row(*row_content)

# Encapsulate the dashboard in a panel
panel_content = Table.grid(expand=True)
panel_content.add_row(Panel(Text("Modern Faux Dashboard", justify="center", style="bold cyan"), padding=(0, 1)))
panel_content.add_row(dashboard)

# Create the main panel
main_panel = Panel(
    panel_content,
    title="Full-Screen Dashboard",
    style="on black",
    width=total_width,
    height=total_height,
    padding=0,
)

# Render the panel
console.print(main_panel)

