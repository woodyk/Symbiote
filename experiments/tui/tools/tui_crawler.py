#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: tui_crawler.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 09:30:11
# Modified: 2024-11-26 09:33:11
#!/usr/bin/env python3
#
# tui_crawler.py

import unicodedata
from rich.panel import Panel
from rich.box import Box
from rich.console import Console
from wcwidth import wcwidth

"""Defines characters to render boxes.

┌─┬┐ top
│ ││ head
├─┼┤ head_row
│ ││ mid
├─┼┤ row
├─┼┤ foot_row
│ ││ foot
└─┴┘ bottom

Args:
    box (str): Characters making up box.
    ascii (bool, optional): True if this box uses ascii characters only. Default is False.
"""

def box_parse_validate(box: str, *, ascii: bool = False) -> None:
    self._box = box
    self.ascii = ascii
    line1, line2, line3, line4, line5, line6, line7, line8 = box.splitlines()
    # top
    self.top_left, self.top, self.top_divider, self.top_right = iter(line1)
    # head
    self.head_left, _, self.head_vertical, self.head_right = iter(line2)
    # head_row
    (
        self.head_row_left,
        self.head_row_horizontal,
        self.head_row_cross,
        self.head_row_right,
    ) = iter(line3)

    # mid
    self.mid_left, _, self.mid_vertical, self.mid_right = iter(line4)
    # row
    self.row_left, self.row_horizontal, self.row_cross, self.row_right = iter(line5)
    # foot_row
    (
        self.foot_row_left,
        self.foot_row_horizontal,
        self.foot_row_cross,
        self.foot_row_right,
    ) = iter(line6)
    # foot
    self.foot_left, _, self.foot_vertical, self.foot_right = iter(line7)
    # bottom
    self.bottom_left, self.bottom, self.bottom_divider, self.bottom_right = iter(
        line8
    )

def create_box_template(line_horizontal: str, line_vertical: str, corner_top_left: str,
                        corner_top_right: str, corner_bottom_left: str, corner_bottom_right: str,
                        divider_horizontal: str, divider_vertical: str, cross: str) -> str:
    """
    Create a box template with the specified characters.

    Args:
        line_horizontal (str): Character for horizontal lines.
        line_vertical (str): Character for vertical lines.
        corner_top_left (str): Character for the top-left corner.
        corner_top_right (str): Character for the top-right corner.
        corner_bottom_left (str): Character for the bottom-left corner.
        corner_bottom_right (str): Character for the bottom-right corner.
        divider_horizontal (str): Character for horizontal dividers.
        divider_vertical (str): Character for vertical dividers.
        cross (str): Character for intersections.

    Returns:
        str: A formatted string that represents the box template.
    """
    return (
        f"{corner_top_left}{line_horizontal*3}{divider_horizontal}{corner_top_right}\n"  # Top row
        f"{line_vertical}   {line_vertical}{line_vertical}\n"                           # Header row
        f"{divider_vertical}{line_horizontal*3}{cross}{divider_vertical}\n"            # Header divider
        f"{line_vertical}   {line_vertical}{line_vertical}\n"                           # Mid row
        f"{divider_vertical}{line_horizontal*3}{cross}{divider_vertical}\n"            # Body divider
        f"{divider_vertical}{line_horizontal*3}{cross}{divider_vertical}\n"            # Foot divider
        f"{line_vertical}   {line_vertical}{line_vertical}\n"                           # Footer row
        f"{corner_bottom_left}{line_horizontal*3}{divider_horizontal}{corner_bottom_right}\n"  # Bottom row
    )

def is_renderable_in_terminal(char):
    """
    Check if a character is renderable in the terminal.

    Args:
        char (str): Character to check.

    Returns:
        bool: True if the character is renderable, False otherwise.
    """
    try:
        width = wcwidth(char)
        if width < 0:  # Unrenderable characters
            return False
        print(f"Renderable character: {repr(char)}")  # Debugging
        return True
    except Exception:
        return False


def create_box_class(line_char, corner_char):
    """
    Create a Box instance with custom characters for rendering.

    Args:
        line_char (str): Character to use for lines.
        corner_char (str): Character to use for corners.

    Returns:
        Box: Custom Box instance.
    """
    box_str = (
        f"{corner_char}{line_char*5}{corner_char}\n"
        f"{line_char}     {line_char}\n"
        f"{line_char}     {line_char}\n"
        f"{corner_char}{line_char*5}{corner_char}\n"
    )
    return Box(box_str)


def render_sample_panel(console, box, line_char, corner_char, line_name, corner_name):
    """
    Render a sample panel using the given box and character.

    Args:
        console (Console): Console instance for rendering.
        box (Box): Box instance for rendering.
        line_char (str): Character used for lines.
        corner_char (str): Character used for corners.
        line_name (str): Unicode name of the line character.
        corner_name (str): Unicode name of the corner character.
    """
    title = f"Panel: Line='{line_char}' ({line_name}), Corner='{corner_char}' ({corner_name})"
    panel = Panel(
        "This panel tests line and corner combinations\n"
        "to identify suitable characters for TUIs.",
        box=box,
        title=title,
        title_align="center",
        padding=(1, 2),
    )
    console.print(panel)


def crawl_unicode_and_test_elements(start=0x2500, end=0x259F, limit=10):
    """
    Crawl through Unicode characters, create box patterns, and render panels.

    Args:
        start (int): Starting Unicode codepoint.
        end (int): Ending Unicode codepoint.
        limit (int): Maximum number of tests to run.
    """
    console = Console()
    count = 0
    line_char_candidates = []
    corner_char_candidates = []

    # Collect valid characters for lines and corners
    for codepoint in range(start, end + 1):
        char = chr(codepoint)
        if is_renderable_in_terminal(char):
            if unicodedata.category(char).startswith("L") or unicodedata.category(char) in ["So", "Sm"]:
                line_char_candidates.append(char)
            elif unicodedata.category(char) in ["Sc", "Pd", "Po"]:
                corner_char_candidates.append(char)

    # Iterate through line and corner combinations
    for line_char in line_char_candidates:
        for corner_char in corner_char_candidates:
            if count >= limit:
                return
            line_name = unicodedata.name(line_char, "Unknown Name")
            corner_name = unicodedata.name(corner_char, "Unknown Name")
            try:
                box = create_box_class(line_char, corner_char)
                render_sample_panel(console, box, line_char, corner_char, line_name, corner_name)
                count += 1
            except Exception as e:
                pass
                #console.print(f"[red]Error rendering panel for {repr(line_char)} and {repr(corner_char)}: {e}[/red]")


if __name__ == "__main__":
    # Set range to box-drawing characters for focused testing
    start_unicode = 0x0000
    end_unicode = 0x259F
    test_limit = 20  # Limit for tests

    print("Generating box patterns and testing line/corner combinations...")
    crawl_unicode_and_test_elements(start=start_unicode, end=end_unicode, limit=test_limit)

