#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: tui-char.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 10:15:59
# Modified: 2024-11-26 16:26:50
#!/usr/bin/env python3
#
# tui-char.py

def get_tui_characters():
    """Categorize and collect Unicode characters useful for TUI design."""
    """
    categories = {
        "Box Drawing": (0x2500, 0x257F),  # Basic box-drawing characters
        "Block Elements": (0x2580, 0x259F),  # Full block, half block, and other large shapes
        "Arrows": (0x2190, 0x21FF),  # Arrows for navigation and separators
        "Geometric Shapes": (0x25A0, 0x25FF),  # Squares, circles, and triangles
        "Miscellaneous Symbols": (0x2600, 0x26FF),  # Includes stars, crosses, and weather symbols
        "Dingbats": (0x2700, 0x27BF),  # Decorative symbols, large bullets, and stars
        "Mathematical Operators": (0x2200, 0x22FF),  # Operators with structured shapes
        "Latin-1 Supplement": (0x0080, 0x00FF),  # Accented characters and special punctuation
        "Enclosed Alphanumeric Supplement": (0x1F100, 0x1F1FF),  # Circled numbers and letters
        "Supplemental Arrows": (0x2B00, 0x2BFF),  # Extended arrows and related symbols
        "Braille Patterns": (0x2800, 0x28FF),  # Compact patterns useful for grids
        "CJK Ideographs": (0x4E00, 0x9FFF),  # Dense, square-shaped ideographs
        "Runic": (0x16A0, 0x16FF),  # Old Germanic symbols with structured shapes
        "Ogham": (0x1680, 0x169F),  # Straight-line and compact symbols
        "Cherokee": (0x13A0, 0x13FF),  # Bold and angular shapes
        "Khmer": (0x1780, 0x17FF),  # Curved and decorative shapes
        "Sinhala": (0x0D80, 0x0DFF),  # Rounded and compact glyphs
        "Devanagari": (0x0900, 0x097F),  # Some straight-line features
        "Tamil": (0x0B80, 0x0BFF),  # Curved, compact letters
        "Ethiopic": (0x1200, 0x137F),  # Bold and square-like symbols
        "Cyrillic": (0x0400, 0x04FF),  # Structured, bold glyphs
        "Greek": (0x0370, 0x03FF),  # Decorative and symmetrical letters
        "Miscellaneous Technical Symbols": (0x2300, 0x23FF),  # Buttons, control symbols, and drawing elements
        "Letterlike Symbols": (0x2100, 0x214F),  # Circled and stylized letters
        "Superscripts and Subscripts": (0x2070, 0x209F),  # Compact characters
        "Halfwidth and Fullwidth Forms": (0xFF00, 0xFFEF),  # Wider or narrower versions of common characters
        "Private Use Area": (0xE000, 0xF8FF),  # Reserved for custom characters
        "Mathematical Alphanumeric Symbols": (0x1D400, 0x1D7FF),  # Bold and italic symbols
        "Large Blocks and Panels": (0x2B50, 0x2B7F),  # Stars and other large shapes
        "CJK Compatibility Forms": (0xFE30, 0xFE4F),  # Wide punctuation and brackets
        "Miscellaneous Pictographs": (0x1F300, 0x1F5FF),  # Decorative emojis, weather symbols, etc.
    }
    """
    categories = {
        "All": (0x0000, 0xFFFD),
    }

    tui_characters = []
    for category, (start, end) in categories.items():
        for i in range(start, end +1):
            if chr(i).isprintable():
                tui_characters.extend(chr(i))
        #tui_characters.extend(chr(i) for i in range(start, end + 1) if chr(i).isprintable())
    return tui_characters

def print_table(characters, single_line=False, columns=30, spacing=4):
    """Print characters in a table format with specified columns and spacing."""
    if single_line is True:
        end_char = ""
    else:
        end_char = "\n"

    for i, char in enumerate(characters):
        print(char.ljust(spacing), end="")
        if (i + 1) % columns == 0:
            print(end=end_char)  # New line after reaching the column count
    if len(characters) % columns != 0:
        print(end=end_char)  # Final newline if the last row is incomplete

def main():
    print("TUI-Friendly Unicode Characters:\n")
    tui_characters = get_tui_characters()
    print_table(tui_characters, single_line=False, columns=28, spacing=2)

if __name__ == "__main__":
    main()

