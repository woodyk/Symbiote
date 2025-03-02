#!/usr/bin/env python3
#
# tuit-classify.py
"""
TUI Character Classification and Template Matching Script

This script is designed to classify terminal characters into various categories
relevant for creating text-based Terminal User Interfaces (TUIs). It also maps
these characters to predefined templates to identify which characters are suitable
for rendering patterns such as tables, forms, banners, and panels.

### Categories of Characters
The script classifies characters into the following categories:
1. **Horizontal Line Characters**: Characters that form horizontal lines
   (e.g., '─', '━', '═').
2. **Vertical Line Characters**: Characters that form vertical lines
   (e.g., '│', '┃', '║').
3. **Corner Characters**:
   - Top Left Corner (e.g., '┌', '╭', '┏').
   - Top Right Corner (e.g., '┐', '╮', '┓').
   - Bottom Left Corner (e.g., '└', '╰', '┗').
   - Bottom Right Corner (e.g., '┘', '╯', '┛').
4. **Junctions**: Characters that connect vertical and horizontal lines
   (e.g., '┬', '┼', '┴', '╋').
5. **Symbols**: Decorative or symbolic characters used in TUIs
   (e.g., '○', '●', '═').

### Template Matching
The script includes predefined TUI templates (e.g., MIXED_STYLE, CURVED_BRACKET)
that represent common layouts for tables, panels, and other UI components. The
script maps the classified characters to these templates to identify:
- Which characters are used in the templates.
- How the characters can be reused or replaced in similar patterns.

### How It Works
1. **Classification**:
   - Characters are classified into their respective categories based on predefined
     sets of characters relevant to TUIs.
2. **Template Matching**:
   - The classified characters are mapped against predefined templates, and matches
     are stored for each template.
3. **Output**:
   - The script outputs:
     - A breakdown of characters by category.
     - A mapping of characters to templates.

### Usage
1. Run the script to classify and match default characters:
   ```bash
   python tui_character_analysis.py
"""


from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# Define categories
categories = {
    "horizontal": ["─", "━", "┈", "┉", "═"],
    "vertical": ["│", "┃", "┊", "┋", "║"],
    "top_left_corner": ["┌", "╭", "┏"],
    "top_right_corner": ["┐", "╮", "┓"],
    "bottom_left_corner": ["└", "╰", "┗"],
    "bottom_right_corner": ["┘", "╯", "┛"],
    "junctions": ["┬", "┼", "┴", "┣", "╋", "╞", "╡"],
    "symbols": ["○", "●", "■", "□", "═"],
}

# Templates to map against
templates = {
    "MIXED_STYLE": [
        "╭─┬─╮",
        "├─┼─┤",
        "╰─┴─╯"
    ],
    "CURVED_BRACKET": [
        "╭───╮",
        "╞═══╡",
        "╰───╯"
    ],
    "DOTTED_LINE": [
        "┈┈┈┈┈",
        "┊   ┊",
    ],
    "DASHED_LINE": [
        "┉┉┉┉┉",
        "┋   ┋",
    ],
    "MIXED_ROUNDED": [
        "╭○○○╮",
        "├○○○┤",
        "╰○○○╯"
    ],
    "MIXED_HEAVY_THIN": [
        "┏━┳━┓",
        "┣━╋━┫",
        "┗━┻━┛"
    ],
}

def classify_characters():
    """Classify characters into categories."""
    classified = defaultdict(list)

    for category, chars in categories.items():
        for char in chars:
            classified[category].append(char)

    return classified

def match_template(classified):
    """Map classified characters to templates."""
    matches = defaultdict(list)

    for template_name, rows in templates.items():
        for row in rows:
            for char in row:
                for category, chars in classified.items():
                    if char in chars:
                        matches[template_name].append(char)
    return matches

def main():
    # Classify characters into categories
    classified = classify_characters()

    # Map classified characters to templates
    matches = match_template(classified)

    print("Character Classification:")
    for category, chars in classified.items():
        print(f"{category}: {''.join(chars)}")

    print("\nTemplate Matches:")
    for template, chars in matches.items():
        print(f"{template}: {''.join(set(chars))}")

if __name__ == "__main__":
    main()

