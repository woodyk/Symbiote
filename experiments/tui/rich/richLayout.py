#!/usr/bin/env python3
#
# richLayout.py

from rich.layout import Layout
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import time

# Initialize Console
console = Console()

# Get console dimensions
console_width = console.size.width
console_height = console.size.height - 4

# Ensure minimum console size for layout demonstration
if console_width < 40 or console_height < 20:
    console.print("[red bold]Please resize your terminal to at least 40x20 for this demo.[/red bold]")
    exit()

def clean():
    time.sleep(2)
    console.clear()

# Example 1: Basic Layout
console.rule("Example 1: Basic Layout")
layout = Layout(name="root")
console.print(layout)
clean()

# Example 2: Column and Row Splits
console.rule("Example 2: Column and Row Splits")
layout.split_column(
    Layout(name="header", size=3),
    Layout(name="body"),
    Layout(name="footer", size=3),
)
layout["body"].split_row(
    Layout(name="left", ratio=2),
    Layout(name="right", ratio=1),
)
console.print(layout)
clean()


# Example 3: Adding Renderables
console.rule("Example 3: Adding Renderables")
layout["header"].update(Panel("Header Section", style=""))
layout["footer"].update(Panel("Footer Section", style=""))
layout["left"].update(
    Panel("Left Pane Content", style="")
)
layout["right"].update(
    Panel("Right Pane Content", style="")
)
console.print(layout)
clean()

# Example 4: Fixed Sizes and Ratios
console.rule("Example 4: Fixed Sizes and Ratios")
layout["header"].size = 5
layout["left"].ratio = 3
layout["right"].ratio = 1
console.print(layout)
clean()

# Example 5: Visibility Control
console.rule("Example 5: Visibility Control")
layout["footer"].visible = False
console.print(layout)
clean()
layout["footer"].visible = True

# Example 6: Dynamic Updates
console.rule("Example 6: Dynamic Updates")
table = Table(title="Sample Table")
table.add_column("Name", justify="left")
table.add_column("Value", justify="right")
table.add_row("Item 1", "Value 1")
table.add_row("Item 2", "Value 2")
layout["right"].update(table)
console.print(layout)
clean()

# Example 7: Nested Splits
console.rule("Example 7: Nested Splits")
layout["left"].split_column(
    Layout(name="left_upper", ratio=1),
    Layout(name="left_lower", ratio=2),
)
layout["left_upper"].update(Panel("Upper Section"))
layout["left_lower"].update(Panel("Lower Section"))
console.print(layout)
clean()

# Example 8: Tree Visualization
console.rule("Example 8: Tree Visualization")
console.print(layout.tree)
clean()

# Example 9: Fullscreen Example
console.rule("Example 9: Fullscreen Example")
fullscreen_layout = Layout(name="fullscreen")
fullscreen_layout.split_column(
    Layout(Panel("Top Section", style=""), size=3),
    Layout(name="main"),
    Layout(Panel("Bottom Section", style=""), size=3),
)
fullscreen_layout["main"].split_row(
    Layout(Panel("Left", style=""), ratio=2),
    Layout(Panel("Right", style=""), ratio=1),
)
console.print(fullscreen_layout)

# Example 10: Adaptive Layout
console.rule("Adaptive Layout")
layout = Layout(name="root")

# Split the layout proportionally based on console height
layout.split_column(
    Layout(name="header", size=3),  # Fixed size for header
    Layout(name="body", ratio=3),  # Ratio-based body
    Layout(name="footer", size=3),  # Fixed size for footer
)

# Split the body horizontally
layout["body"].split_row(
    Layout(name="left", ratio=2),
    Layout(name="right", ratio=1),
)

# Adding content dynamically
layout["header"].update(Panel(f"Header Section (Width: {console_width})", style="cyan"))
layout["footer"].update(Panel(f"Footer Section (Width: {console_width})", style="magenta"))

layout["left"].update(Panel("Left Pane Content", style="green"))
layout["right"].update(Panel("Right Pane Content", style="yellow"))

# Nested splits in the left pane based on remaining height
remaining_height = console_height - 6  # Subtracting header and footer sizes
if remaining_height > 6:  # Only nest if there's enough space
    layout["left"].split_column(
        Layout(name="left_upper", size=remaining_height // 2),
        Layout(name="left_lower", size=remaining_height // 2),
    )
    layout["left_upper"].update(Panel("Upper Section"))
    layout["left_lower"].update(Panel("Lower Section"))

# Table example in the right pane
table = Table(title="Sample Table")
table.add_column("Name", justify="left")
table.add_column("Value", justify="right")
table.add_row("Item 1", "Value 1")
table.add_row("Item 2", "Value 2")
layout["right"].update(table)

# Render the layout
console.print(layout)
clean()

# Display the layout tree
console.rule("Layout Tree")
console.print(layout.tree)
