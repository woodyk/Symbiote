#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: panel_test.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 09:28:46
#!/usr/bin/env python3
#
# panel_test.py

DASHED_ROUNDED = (
    "╭┄┄┄╮\n"
    "┆   ┆\n"
    "├┄┄┄┤\n"
    "┆   ┆\n"
    "├┄┄┄┤\n"
    "├┄┄┄┤\n"
    "┆   ┆\n"
    "╰┄┄┄╯\n"
)


def create_box_template(corner_char: str, line_char: str) -> str:
    """
    Create a box template using a corner and line character.

    Args:
        corner_char (str): Character for corners.
        line_char (str): Character for horizontal and vertical lines.

    Returns:
        str: A formatted string that represents the box template.
    """
    template = (
        f"{corner_char}{line_char*3}{corner_char}\n"  # Top row
        f"{line_char}   {line_char}{line_char}\n"               # Header row
        f"{corner_char}{line_char*3}{corner_char}\n"  # Header divider
        f"{line_char}   {line_char}{line_char}\n"               # Mid row
        f"{corner_char}{line_char*3}{corner_char}\n"  # Body divider
        f"{corner_char}{line_char*3}{corner_char}\n"  # Foot divider
        f"{line_char}   {line_char}{line_char}\n"               # Footer row
        f"{corner_char}{line_char*3}{corner_char}\n"  # Bottom row
    )
    print(template)
    return template


def generate_panel(corner_char: str, line_char: str = None, panel_content: str = "Custom Panel"):
    """
    Generate and display a rich panel with the given corner and line characters.

    Args:
        corner_char (str): Character for corners.
        line_char (str): Character for lines. Defaults to corner_char if None.
        panel_content (str): Content of the panel.
    """
    console = Console()
    line_char = line_char or corner_char  # Default to corner_char if line_char is not provided

    try:
        # Generate the box template
        box_template = create_box_template(corner_char, line_char)

        # Create a Box instance
        custom_box = Box(box_template)

        # Render the panel
        panel = Panel(
            panel_content,
            title=f"Corner='{corner_char}', Line='{line_char}'",
            title_align="center",
            box=custom_box,
            padding=(1, 2),
            width=40,  # Set panel width
        )
        console.print(panel)

    except Exception as e:
        console.print(f"[red]Error creating panel: {e}[/red]")


if __name__ == "__main__":
    # Get user input for the corner and line characters
    #corner = input("Enter the corner character: ")
    #line = input("Enter the line character (press Enter to use the corner character): ")
    create_box_template("ꄨ", "⺫")

    # Generate the panel
    #generate_panel(corner, line if line.strip() else None)

