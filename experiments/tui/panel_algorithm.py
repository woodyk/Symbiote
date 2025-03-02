#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: panel_algorithm.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 13:24:55
# Modified: 2024-11-26 19:30:13


"""
### Generalized TUI Rendering Function with Template System

This version introduces a **template system** for defining and managing
character configurations for TUI panels, boxes, or tables. Each template
defines the characters for various render locations (e.g., borders,
corners) and allows customization while ensuring proper alignment with
mixed-width characters.

### Explanation of the Algorithm:
1. **Templates**:
   - A dictionary (`TUI_TEMPLATES`) stores predefined character sets for 
     different styles (e.g., "default", "bold").
   - Each character set specifies what characters to use for corners, 
     borders, and junctions.

2. **Character Width Measurement**:
   - The `wcwidth` library is used to measure the visual width of each 
     character, including multi-character strings.
   - The largest character width is determined and used as a baseline for 
     alignment.

3. **Auto-Scaling**:
   - Characters are scaled to match the width of the largest character in 
     the template for consistent alignment.

4. **Dynamic Dimensions**:
   - The width and height are auto-calculated based on the terminal size, 
     padding, and content.

5. **Content Alignment**:
   - Content is padded with spaces to center it within the calculated inner 
     width of the panel.

6. **Rendering**:
   - Borders, title, and content rows are assembled and printed in order.
"""

import os
from wcwidth import wcwidth  # For accurate character width measurements

# Default templates for TUI components
TUI_TEMPLATES = {
    "default": {
        "top_left": "┌",
        "top_right": "┐",
        "bottom_left": "└",
        "bottom_right": "┘",
        "horizontal": "─",
        "vertical": "│",
        "cross": "┼",
        "left": "├",
        "right": "┤",
    },
}

import os
from wcwidth import wcwidth

def calculate_padding(inner_width, text_length):
    """
    Calculate left and right padding to center text.

    Args:
        inner_width (int): The width of the content area.
        text_length (int): The length of the text to be centered.

    Returns:
        tuple: Padding for left and right (left_padding, right_padding).
    """
    padding_left = (inner_width - text_length) // 2
    padding_right = inner_width - text_length - padding_left
    return padding_left, padding_right

def render_panel(title=None, content=None, template=None, width=None, padding=2):
    """
    Renders a TUI panel with title and content.

    Args:
        title (str): Title text to be displayed.
        content (str): Content text for the panel.
        template (dict): Template for border characters.
        width (int): Panel width. If None, auto-calculates based on content.
        padding (int): Space around content inside the panel.

    Returns:
        str: Rendered panel as a string.
    """
    template = template or {
        "top_left": "┌",
        "top_right": "┐",
        "bottom_left": "└",
        "bottom_right": "┘",
        "horizontal": "─",
        "vertical": "│",
    }

    def char_width(char):
        return wcwidth(char)

    # Validate and measure character widths
    max_char_width = max(char_width(c) for c in template.values())
    scaled_template = {k: v * max(1, max_char_width // char_width(v)) for k, v in template.items()}

    # Auto-calculate dimensions
    terminal_width = os.get_terminal_size().columns
    content_lines = content.splitlines() if content else []
    content_width = max(len(line) for line in content_lines) if content else 0
    width = width or min(terminal_width, content_width + padding * 2 + 2)
    inner_width = width - 2 * max_char_width

    # Render rows
    top_border = f"{scaled_template['top_left']}{scaled_template['horizontal'] * (inner_width // max_char_width)}{scaled_template['top_right']}"
    bottom_border = f"{scaled_template['bottom_left']}{scaled_template['horizontal'] * (inner_width // max_char_width)}{scaled_template['bottom_right']}"

    title_row = ""
    if title:
        padding_left, padding_right = calculate_padding(inner_width, len(title))
        title_row = f"{scaled_template['vertical']}{' ' * padding_left}{title}{' ' * padding_right}{scaled_template['vertical']}"

    content_rows = []
    for line in content_lines:
        padding_left, padding_right = calculate_padding(inner_width, len(line))
        content_rows.append(f"{scaled_template['vertical']}{' ' * padding_left}{line}{' ' * padding_right}{scaled_template['vertical']}")

    # Add empty rows for padding
    empty_row = f"{scaled_template['vertical']}{' ' * inner_width}{scaled_template['vertical']}"
    padding_rows = [empty_row] * padding

    # Assemble final panel
    rows = [top_border] + padding_rows + ([title_row] if title else []) + content_rows + padding_rows + [bottom_border]
    return "\n".join(rows)

# Example usage
if __name__ == "__main__":
    print(render_panel(title="Hello", content="This is a test.\nWith multiple lines."))

