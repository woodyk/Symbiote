#!/usr/bin/env python3
#
# AutoDash.py

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.pretty import Pretty
from rich import box

def generate_dashboard(name: str, layout: dict):
    console = Console()
    grid = layout.get('grid', [])
    rows = layout.get('rows', [])
    content = layout.get('content', [])
    default_style = "white"
    default_box = box.SQUARE
    default_title = "" 

    dashboard = []

    for i, row in enumerate(grid):
        row_panels = []
        total_specified_width = 0

        for w in row:
            if w > 0:
                total_specified_width += w
        unspecified_count = row.count(0)
        if unspecified_count > 0:
            auto_width = (100 - total_specified_width) / unspecified_count
        else:
            auto_width = None 

        for j, width in enumerate(row):
            title = default_title 
            style = border_style = default_style 
            if i < len(rows) and j < len(rows[i]):
                row_value = rows[i][j]
                if isinstance(row_value, str):
                    title = row_value
                elif row_value is True:
                    pass  # Border with no title
                elif row_value is None:
                    border_style = "none" 
                elif row_value is False:
                    title = default_title 
                    border_style = "none" 

            # Get panel content
            panel_content = ''
            if i < len(content) and j < len(content[i]):
                content_value = content[i][j]
                if callable(content_value):
                    panel_content = content_value()
                elif isinstance(content_value, (dict, list, set)):
                    panel_content = Pretty(content_value)
                else:
                    panel_content = content_value
            
            if width > 0:
                panel_width = int(console.width * width / 100)
            elif width == 0:
                panel_width = None

            # Create panel
            panel = Panel(
                panel_content,
                title=title,
                border_style=border_style,
                style=style,
                width=panel_width,
                box=default_box,
                expand=True,
            )
            row_panels.append(panel)

        # Create a Columns object for the row
        row_layout = Columns(row_panels, expand=True)
        dashboard.append(row_layout)

    # Render the dashboard
    console.print(*dashboard)

# Example usage
generate_dashboard("My Dashboard", {
    "grid": [
        [0, 0, 0, 0],
        [25, 75],
        [0, 30]
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

