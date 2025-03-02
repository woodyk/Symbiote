#!/usr/bin/env python3
#
# richColumns.py

from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

# Initialize Console
console = Console()

# Create some renderable objects
panels = [
    Panel(f"Panel {i+1}", title=f"Title {i+1}") for i in range(9)
]

# Example 1: Default Columns
console.rule("Default Columns Example")
columns_default = Columns(panels)
console.print(columns_default)

# Example 2: Columns with Padding
console.rule("Columns with Padding Example")
columns_with_padding = Columns(panels, padding=(1, 2))
console.print(columns_with_padding)

# Example 3: Specified Column Width
console.rule("Columns with Specified Width Example")
columns_with_width = Columns(panels, width=20)
console.print(columns_with_width)

# Example 4: Expanded Columns
console.rule("Expanded Columns Example")
columns_expanded = Columns(panels, expand=True)
console.print(columns_expanded)

# Example 5: Equal Sized Columns
console.rule("Equal Sized Columns Example")
columns_equal = Columns(panels, equal=True)
console.print(columns_equal)

# Example 6: Column-First Alignment
console.rule("Column-First Alignment Example")
columns_column_first = Columns(panels, column_first=True)
console.print(columns_column_first)

# Example 7: Right-to-Left Columns
console.rule("Right-to-Left Columns Example")
columns_right_to_left = Columns(panels, right_to_left=True)
console.print(columns_right_to_left)

# Example 8: Aligned Columns
console.rule("Aligned Columns Example (Center)")
columns_aligned_center = Columns(panels, align="center")
console.print(columns_aligned_center)

# Example 9: Columns with Title
console.rule("Columns with Title Example")
columns_with_title = Columns(panels, title="Neatly Arranged Panels")
console.print(columns_with_title)

# Example 10: Adding Renderables Dynamically
console.rule("Dynamically Adding Renderables Example")
dynamic_columns = Columns(title="Dynamic Columns Example")
for i in range(5):
    dynamic_columns.add_renderable(Panel(f"Dynamic Panel {i+1}"))
console.print(dynamic_columns)

# Example 11: Mixed Renderables
console.rule("Mixed Renderables Example")
mixed_renderables = Columns(
    [
        Panel("This is a panel"),
        Text("This is just text"),
        Panel("Another panel"),
        "This is a plain string",
    ],
    title="Mixed Renderables",
    align="center",
    width=30,
)
console.print(mixed_renderables)

# Summary of features
console.rule("Summary of Columns Features")
summary_text = Text.from_markup(
    "[bold cyan]1. Default Columns:[/] Arrange items in auto-detected columns.\n"
    "[bold green]2. Padding:[/] Add space around renderables.\n"
    "[bold yellow]3. Specified Width:[/] Fix the width of columns.\n"
    "[bold magenta]4. Expanded Columns:[/] Columns expand to full width.\n"
    "[bold blue]5. Equal Sized Columns:[/] All columns have equal size.\n"
    "[bold red]6. Column-First:[/] Align items vertically first.\n"
    "[bold white]7. Right-to-Left:[/] Start columns from the right.\n"
    "[bold orange]8. Alignment:[/] Align text in columns (left, center, right).\n"
    "[bold purple]9. Title:[/] Add a title to the columns.\n"
    "[bold gray]10. Dynamic Renderables:[/] Add renderables dynamically.\n"
    "[bold teal]11. Mixed Renderables:[/] Mix various renderable types."
)
console.print(Panel(summary_text, title="Summary"))

