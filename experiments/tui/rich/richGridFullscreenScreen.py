#!/usr/bin/env python3
#
# richGridFullscreenScreen.py

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
import time

# Initialize Console
console = Console()

# Full-Screen Grid Rendering with `Console.screen`
with console.screen():
    # Define grid structure
    full_screen_grid = Table.grid(expand=True)
    
    # Number of rows and columns for the grid
    rows, cols = 6, 3  # Customize these as needed
    
    # Add columns dynamically based on the grid size
    for col in range(cols):
        full_screen_grid.add_column(justify="center", ratio=1)
    
    # Add rows dynamically with cell content
    for row in range(rows):
        row_content = [
            f"[bold magenta]Cell {row + 1}-{col + 1}[/bold magenta]"
            for col in range(cols)
        ]
        full_screen_grid.add_row(*row_content)
    
    # Add a footer panel with dynamic terminal information
    terminal_size = console.size
    footer_panel = Panel(
        Text(
            f"Terminal Size: {terminal_size.width}x{terminal_size.height}",
            justify="center",
            style="dim cyan",
        ),
        style="bold green",
    )
    
    # Display the grid and footer
    console.print(full_screen_grid)
    console.print(footer_panel)
    
    # Wait for 5 seconds
    time.sleep(5)

