#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: ok.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 09:25:24
#!/usr/bin/env python3
#
# ok.py

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.pretty import Pretty

def generate_dashboard(name: str, layout: dict):
    console = Console()
    grid_layout = layout.get('grid', [])
    rows_layout = layout.get('rows', [])
    content_layout = layout.get('content', [])
    
    # Initialize the main grid
    main_grid = Table.grid(expand=True)
    
    # Calculate the maximum number of columns
    max_columns = max(len(row) for row in grid_layout)
    
    # Add columns to the main grid
    for _ in range(max_columns):
        main_grid.add_column(ratio=1)
    
    # Build the grid row by row
    for row_index, row in enumerate(grid_layout):
        row_cells = []
        for col_index, width in enumerate(row):
            # Get the panel title and border settings
            title = None
            border = True
            if row_index < len(rows_layout) and col_index < len(rows_layout[row_index]):
                row_value = rows_layout[row_index][col_index]
                if isinstance(row_value, str):
                    title = row_value
                elif row_value is True:
                    pass  # Border with no title
                elif row_value is None:
                    border = False
                elif row_value is False:
                    border = False
            else:
                border = False  # Default border setting
            
            # Get the panel content
            panel_content = ''
            if row_index < len(content_layout) and col_index < len(content_layout[row_index]):
                content_value = content_layout[row_index][col_index]
                if callable(content_value):
                    panel_content = content_value()
                elif isinstance(content_value, (dict, list, set)):
                    panel_content = Pretty(content_value)
                else:
                    panel_content = content_value
            else:
                panel_content = ''
            
            # Create the panel
            panel = Panel(
                panel_content,
                title=title,
                border_style='white' if border else None,
                padding=(1, 2),
            )
            
            # Append the panel to the row cells
            row_cells.append(panel)
        
        # Ensure the row has the correct number of columns
        while len(row_cells) < max_columns:
            row_cells.append(Panel('', border_style=None))
        
        # Add the row to the main grid
        main_grid.add_row(*row_cells)
    
    # Add a header panel if a name is provided
    if name:
        header_panel = Panel(Text(name, justify="center", style="bold magenta"), style="on cyan")
        # Create an overall grid to include the header and main grid
        overall_grid = Table.grid(expand=True)
        overall_grid.add_column()
        overall_grid.add_row(header_panel)
        overall_grid.add_row(main_grid)
        console.print(overall_grid)
    else:
        console.print(main_grid)

# Example usage
generate_dashboard("Advanced Console Layout", {
    "grid": [
        [0, 0, 0, 0],
        [25, 75],
        [0, 30]
    ],
    "rows": [
        ["Section 1", "Section 2", "Section 3", "Section 4"],
        [True, "Numbers"],
        [False, False]
    ],
    "content": [
        [
            "Content for Section 1",
            "Content for Section 2",
            "Content for Section 3",
            "Content for Section 4"
        ],
        [
            lambda: "[bold red]Dynamic Content[/bold red]",
            {"key": "value", "number": 42}
        ],
        [
            "Additional Information",
            [1, 2, 3, 4, 5]
        ]
    ]
})

