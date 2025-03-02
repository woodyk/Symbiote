#!/usr/bin/env python3
#
# richGridFullscreenFill.py

from rich.console import Console
from rich.table import Table
from rich.text import Text

# Initialize Console
console = Console()

# Example: Dynamic Full-Screen Grid with Height Padding
console.rule("Dynamic Full-Screen Grid with Height Padding")

# Get terminal dimensions
terminal_size = console.size

# Number of rows and columns for the grid
rows = 6
cols = 3

# Calculate padding based on terminal height
grid_height = rows + 2  # +2 for header/footer or any extra space
padding_top = (terminal_size.height - grid_height) // 2  # Center the grid vertically

# Create a full-screen grid
full_screen_grid = Table.grid(expand=True, padding=(padding_top, 0))

# Add columns dynamically based on the terminal width
for col in range(cols):
    full_screen_grid.add_column(justify="center", ratio=1)

# Add rows dynamically with proportional cell content
for row in range(rows):
    row_content = [f"[bold magenta]Cell {row + 1}-{col + 1}[/bold magenta]" for col in range(cols)]
    full_screen_grid.add_row(*row_content)

# Add additional information about the terminal size
terminal_info = f"Terminal Width: {terminal_size.width}, Height: {terminal_size.height}"
footer_panel = Text(terminal_info, justify="center", style="dim cyan")

# Print the full-screen grid and terminal info
console.print(full_screen_grid)
console.print(footer_panel)

