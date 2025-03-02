#!/usr/bin/env python3
#
# dashgen.py

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.pretty import Pretty
from rich import box


def generate_dashboard(name: str, layout: dict):
    console = Console()
    grid = layout.get("grid", [])
    rows = layout.get("rows", [])
    content = layout.get("content", [])
    default_style = "white"
    default_box = box.SQUARE
    default_title = ""

    dashboard = []
    console_width = console.size.width

    for i, row in enumerate(grid):
        # Create an invisible Table for this row
        table_grid = Table(box=None, show_edge=False, padding=0, expand=True)
        total_specified_width = sum(w for w in row if w > 0)
        unspecified_count = row.count(0)
        auto_width = (100 - total_specified_width) / unspecified_count if unspecified_count > 0 else 0

        # Track total width for validation
        row_total_width = 0

        for j, width in enumerate(row):
            panel_width = None
            if width > 0:
                panel_width = int(console_width * width / 100)
            elif width == 0:
                panel_width = int(console_width * auto_width / 100) if auto_width else None

            # Ensure the total width does not exceed console width
            row_total_width += panel_width if panel_width else 0
            if row_total_width > console_width:
                raise ValueError(f"Row {i+1} exceeds console width. Adjust grid widths.")

            # Configure column
            table_grid.add_column(width=panel_width)

            # Determine title and border behavior
            title = default_title
            border_style = default_style
            if i < len(rows) and j < len(rows[i]):
                row_value = rows[i][j]
                if isinstance(row_value, str):
                    title = row_value
                elif row_value is True:
                    pass  # Border with no title
                elif row_value is None or row_value is False:
                    border_style = "none"

            # Get panel content
            panel_content = ""
            if i < len(content) and j < len(content[i]):
                content_value = content[i][j]
                if callable(content_value):
                    panel_content = content_value()
                elif isinstance(content_value, (dict, list, set)):
                    panel_content = Pretty(content_value)
                else:
                    panel_content = content_value

            # Create the panel
            panel = Panel(
                panel_content,
                title=title,
                border_style=border_style,
                style=default_style,
                width=None,  # Let the table cell manage the width
                box=default_box,
                expand=True,
            )

            # Add panel to the table as a row
            table_grid.add_row(panel)

        # Add the row table to the dashboard
        dashboard.append(table_grid)

    # Render the dashboard with proper vertical spacing
    final_dashboard = Group(*dashboard)
    console.print(Panel(final_dashboard, title=name, box=default_box))


# Example usage
generate_dashboard("My Dashboard", {
    "grid": [
        [25, 25, 25, 25],  # Equal-width panels
        [25, 75],          # Two panels with 25% and 75% width
        [0, 30],           # Flexible panel and fixed 30% width
    ],
    "rows": [
        ["Panel 1", "Panel 2", "Panel 3", "Panel 4"],
        [True, "Numbers"],
        [False, False]
    ],
    "content": [
        ["Content 1", "Content 2", "Content 3", "Content 4"],
        [lambda: "Dynamic Content", {"key": "value"}],
        ["String Content", [1, 2, 3]]
    ]
})

