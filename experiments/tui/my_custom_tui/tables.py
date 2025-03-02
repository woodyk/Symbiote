#!/usr/bin/env python3
#
# tables.py

import shutil

class Box:
    """Defines characters to render boxes."""
    def __init__(self, box_template: str):
        lines = box_template.splitlines()
        # Parse box parts from the template
        self.top_left, self.top, self.top_divider, self.top_right = lines[0]
        self.vertical_left, _, self.vertical, self.vertical_right = lines[1]
        self.head_left, self.head_horizontal, self.head_cross, self.head_right = lines[2]
        self.mid_left, _, self.mid_vertical, self.mid_right = lines[3]
        self.row_left, self.row_horizontal, self.row_cross, self.row_right = lines[4]
        self.foot_row_left, self.foot_row_horizontal, self.foot_row_cross, self.foot_row_right = lines[5]
        self.bottom_left, self.bottom, self.bottom_divider, self.bottom_right = lines[7]

ROUNDED = Box(
    "╭─┬╮\n"
    "│ ││\n"
    "├─┼┤\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "╰─┴╯\n"
)


class TableRenderer:
    def __init__(self, title=None, box=ROUNDED, alignment="left", width=None, padding=4):
        """
        Initialize a new table renderer.

        Args:
            title (str): Title of the table.
            box (Box): The box template for borders and dividers.
            alignment (str): Alignment of the table ('left', 'center', 'right').
            width (int or str): Width of the table (percentage or fixed).
            padding (int): Padding to subtract from full width.
        """
        self.title = title
        self.columns = []
        self.rows = []
        self.box = box
        self.alignment = alignment
        terminal_width = shutil.get_terminal_size().columns
        if isinstance(width, str) and width.endswith("%"):
            self.width = int(terminal_width * (int(width[:-1]) / 100))
        else:
            self.width = width or terminal_width - padding

    def add_column(self, *column_names):
        """Add column names to the table."""
        if not self.columns:
            self.columns = list(column_names)
        else:
            raise ValueError("Columns have already been defined.")

    def add_row(self, *row_data):
        """Add a row to the table."""
        if len(row_data) != len(self.columns):
            raise ValueError("Row data must match the number of columns.")
        self.rows.append(row_data)

    def align_line(self, line):
        """Align a line based on the alignment setting."""
        if self.alignment == "center":
            return line.center(self.width)
        elif self.alignment == "right":
            return line.rjust(self.width)
        elif self.alignment == "left":
            return line.ljust(self.width)
        else:
            raise ValueError("Invalid alignment: choose 'left', 'center', or 'right'.")

    def render(self):
        """Render the table as a string."""
        if not self.columns:
            raise ValueError("No columns defined for the table.")

        # Calculate column widths based on the longest content in each column
        column_widths = [
            max(len(str(item)) for item in [col] + [row[i] for row in self.rows])
            for i, col in enumerate(self.columns)
        ]

        # Ensure total table width fits within specified width
        padding_space = max(self.width - (sum(column_widths) + 3 * len(self.columns) - 1), 0)
        if padding_space > 0:
            column_widths = [width + padding_space // len(column_widths) for width in column_widths]

        box = self.box

        # Top border
        top_border = box.top_left + box.top_divider.join(
            box.top * (width + 2) for width in column_widths
        ) + box.top_right

        # Header row
        header = box.vertical_left + box.vertical.join(
            f" {col.center(width)} " for col, width in zip(self.columns, column_widths)
        ) + box.vertical_right

        # Header/Body separator
        header_separator = box.head_left + box.head_cross.join(
            box.head_horizontal * (width + 2) for width in column_widths
        ) + box.head_right

        # Rows
        rows = []
        for row in self.rows:
            row_line = box.vertical_left + box.vertical.join(
                f" {str(cell).ljust(width)} " for cell, width in zip(row, column_widths)
            ) + box.vertical_right
            rows.append(row_line)

        # Row separators
        row_separator = box.row_left + box.row_cross.join(
            box.row_horizontal * (width + 2) for width in column_widths
        ) + box.row_right

        # Bottom border
        bottom_border = box.bottom_left + box.bottom_divider.join(
            box.bottom * (width + 2) for width in column_widths
        ) + box.bottom_right

        # Title
        total_table_width = sum(column_widths) + 3 * len(self.columns) - 1
        output = []
        if self.title:
            title_line = f" {self.title.center(total_table_width)} "
            output.append(self.align_line(title_line))

        # Assemble the table
        output.append(self.align_line(top_border))
        output.append(self.align_line(header))
        output.append(self.align_line(header_separator))
        for i, row in enumerate(rows):
            output.append(self.align_line(row))
            if i < len(rows) - 1:
                output.append(self.align_line(row_separator))
        output.append(self.align_line(bottom_border))

        # Return the table as a single string
        return "\n".join(output)


# Example Usage
table = TableRenderer(title="Custom Box Table", box=ROUNDED, alignment="center", width="")
table.add_column("Name", "Age", "Occupation")
table.add_row("Alice", 30, "Engineer")
table.add_row("Bob", 25, "Designer")
table.add_row("Charlie", 35, "Teacher")

print(table.render())

