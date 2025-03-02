#!/usr/bin/env python3
#
# richGridLayout.py

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Initialize Console
console = Console()

# Define sections
header = Panel(Text("Advanced Console Layout", justify="center", style="bold magenta"), style="on cyan")
footer = Panel(Text("Footer Information", justify="center", style="bold yellow"), style="on green")

# Define left and right columns for the main layout
left_column = Table.grid(expand=True)
left_column.add_column()
left_column.add_row("[bold cyan]Section 1[/bold cyan]")
left_column.add_row("This is the content of the first section.")
left_column.add_row("[bold cyan]Section 2[/bold cyan]")
left_column.add_row("Another section goes here.")

right_column = Table.grid(expand=True)
right_column.add_column()
right_column.add_row("[bold yellow]Alerts[/bold yellow]")
right_column.add_row("⚠️ [bold red]Critical Issue[/bold red]")
right_column.add_row("[bold green]Everything is running smoothly.[/bold green]")

# Main content area
main_content = Table.grid(expand=True)
main_content.add_column()
main_content.add_row("[bold magenta]Main Content Area[/bold magenta]")
main_content.add_row(
    "Here is where the main information or a dashboard-like content could be displayed. "
    "[dim]You can add tables, charts, or anything else Rich supports.[/dim]"
)

# Create the overall grid layout
layout = Table.grid(expand=True, padding=(1, 2))
layout.add_column(ratio=1)
layout.add_column(ratio=2)
layout.add_column(ratio=1)

layout.add_row(header, header, header)  # Header spans all columns
layout.add_row(left_column, main_content, right_column)  # Main layout
layout.add_row(footer, footer, footer)  # Footer spans all columns

# Print the layout
console.print(layout)

