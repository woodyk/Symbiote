#!/usr/bin/env python3
#
# richTable.py

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table, Column
from rich import box
from rich.align import Align

# Initialize Console
console = Console()

# Example 1: Basic Table
console.rule("Example 1: Basic Table")
table1 = Table(title="Star Wars Movies")

table1.add_column("Released", justify="right", style="cyan", no_wrap=True)
table1.add_column("Title", style="magenta")
table1.add_column("Box Office", justify="right", style="green")

table1.add_row("Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$952,110,690")
table1.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
table1.add_row("Dec 15, 2017", "Star Wars Ep. VIII: The Last Jedi", "$1,332,539,889")
table1.add_row("Dec 16, 2016", "Rogue One: A Star Wars Story", "$1,332,439,889")

console.print(table1)

# Example 2: Table with Custom Borders
console.rule("Example 2: Table with Custom Borders")
table2 = Table(title="Custom Borders", box=box.MINIMAL_DOUBLE_HEAD, border_style="bold magenta")

table2.add_column("Column 1")
table2.add_column("Column 2")

table2.add_row("Data 1", "Data 2")
table2.add_row("Data 3", "Data 4")

console.print(table2)

# Example 3: Table with Row Styles
console.rule("Example 3: Table with Row Styles")
table3 = Table(title="Row Styles", row_styles=["dim", ""])

table3.add_column("Odd Rows")
table3.add_column("Even Rows")

for i in range(1, 11):
    table3.add_row(f"Row {i}" if i % 2 != 0 else "", f"Row {i}" if i % 2 == 0 else "")

console.print(table3)

# Example 4: Table with Line Between Rows
console.rule("Example 4: Table with Line Between Rows")
table4 = Table(title="Lines Between Rows", show_lines=True)

table4.add_column("Name")
table4.add_column("Age")
table4.add_column("Profession")

table4.add_row("Alice", "30", "Engineer")
table4.add_row("Bob", "40", "Doctor")
table4.add_row("Charlie", "25", "Artist")

console.print(table4)

# Example 5: Table with Footer
console.rule("Example 5: Table with Footer")
table5 = Table(title="Footer Example", show_footer=True)

table5.add_column("Category")
table5.add_column("Count", footer="Total: 5")

table5.add_row("Apples", "2")
table5.add_row("Oranges", "3")

console.print(table5)

# Example 6: Grid Layout
console.rule("Example 6: Grid Layout")
grid = Table.grid(expand=True, padding=(0, 1))

grid.add_column()
grid.add_column(justify="right")

grid.add_row("Left Aligned Text", "[bold magenta]Right Aligned Text[/bold magenta]")
grid.add_row("Raising shields", "[green]COMPLETED [green]:heavy_check_mark:")

console.print(grid)

# Example 7: Table with Alignments and Overflow
console.rule("Example 7: Table with Alignments and Overflow")
table7 = Table(title="Alignments and Overflow")

table7.add_column("Left Aligned", justify="left", style="cyan", overflow="fold")
table7.add_column("Center Aligned", justify="center", style="magenta", overflow="ellipsis")
table7.add_column("Right Aligned", justify="right", style="green")

table7.add_row("Short", "This is a long center-aligned text", "12345")
table7.add_row("Another Row", "Another long text that gets ellipsized", "67890")

console.print(table7)

# Example 8: Advanced Table with Multiple Customizations
console.rule("Example 8: Advanced Table with Customizations")
table8 = Table(
    "Name",
    "Role",
    "Department",
    title="Advanced Custom Table",
    caption="This table demonstrates advanced features",
    box=box.DOUBLE_EDGE,
    show_lines=True,
    row_styles=["", "dim"],
    title_justify="center",
    caption_justify="right",
    border_style="bold yellow",
    header_style="bold blue",
    footer_style="italic green",
)

table8.add_column("Name")
table8.add_column("Role")
table8.add_column("Department", footer="End of Data")

table8.add_row("Alice", "Manager", "HR")
table8.add_row("Bob", "Developer", "IT")
table8.add_row("Charlie", "Designer", "Marketing")

console.print(table8)

# Example 9: Empty Table Handling
console.rule("Example 9: Empty Table Handling")
empty_table = Table(title="Empty Table Example")

if empty_table.columns:
    console.print(empty_table)
else:
    console.print("[i]No data available[/i]")

# Example 10: Dynamic Data Table
console.rule("Example 10: Dynamic Data Table")
data = [("Alice", 30, "Engineer"), ("Bob", 40, "Doctor"), ("Charlie", 25, "Artist")]

dynamic_table = Table("Name", "Age", "Profession", title="Dynamic Data Table")
for row in data:
    dynamic_table.add_row(*map(str, row))
console.print(dynamic_table)

# Example 11: Table border styels
console.rule("Example 11: Table Border Sytles")
# List of available border styles
# Sample Data
table_data = [
    ("Alice", "Engineer", "HR"),
    ("Bob", "Doctor", "Medical"),
    ("Charlie", "Artist", "Creative"),
]

# Dynamically retrieve all available border styles from the box module
border_styles = [(name, getattr(box, name)) for name in dir(box) if name.isupper()]

# Generate a table for each border style
for name, style in border_styles:
    console.rule(f"Border Style: {name}", style="bold magenta")
    table = Table(
        "Name",
        "Role",
        "Department",
        title=f"Border Style: {name}",
        box=style,
        border_style="bold cyan",
    )
    for row in table_data:
        table.add_row(*row)
    try:
        console.print(table)
    except:
        pass


