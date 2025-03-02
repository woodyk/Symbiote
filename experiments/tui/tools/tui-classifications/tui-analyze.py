#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: tui-analyze.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 09:36:33
#!/usr/bin/env python3
#

import os
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import string  # Import the string module

def analyze_character(char, font, size=40):
    """Analyze a character and determine its horizontal, vertical, or connective attributes."""
    try:
        bbox = font.getbbox(char)
    except Exception:
        return None  # Skip if rendering fails

    if not bbox:
        return None

    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new("L", (width, height), 255)  # White background
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0], -bbox[1]), char, font=font, fill=0)  # Render character

    # Convert to binary pixel map
    pixels = img.load()
    horizontal_density = [sum(pixels[x, y] < 128 for x in range(width)) for y in range(height)]
    vertical_density = [sum(pixels[x, y] < 128 for y in range(height)) for x in range(width)]

    # Calculate thresholds
    horizontal_line = max(horizontal_density) > 0.7 * width
    vertical_line = max(vertical_density) > 0.7 * height
    corner_or_connector = (
        sum(1 for d in horizontal_density if d > 0.5 * width) > 1 and
        sum(1 for d in vertical_density if d > 0.5 * height) > 1
    )

    return {
        "horizontal": horizontal_line,
        "vertical": vertical_line,
        "corner_or_connector": corner_or_connector,
    }

def catalog_characters(characters, font):
    """Catalog characters into TUI-relevant categories."""
    horizontal_chars = []
    vertical_chars = []
    corner_chars = []
    connectors = []

    for char in characters:
        try:
            attributes = analyze_character(char, font)
            if not attributes:
                continue

            if attributes["horizontal"] and not attributes["vertical"]:
                horizontal_chars.append(char)
            elif attributes["vertical"] and not attributes["horizontal"]:
                vertical_chars.append(char)
            elif attributes["corner_or_connector"]:
                corner_chars.append(char)
            elif attributes["horizontal"] and attributes["vertical"]:
                connectors.append(char)
        except Exception:
            pass  # Ignore errors for individual characters

    return horizontal_chars, vertical_chars, corner_chars, connectors

def detect_patterns(characters):
    """Group characters by proximity for pattern detection."""
    sets = []
    current_set = []

    for i, char in enumerate(characters):
        if i > 0 and ord(char) - ord(characters[i - 1]) > 1:
            if current_set:
                sets.append(current_set)
                current_set = []
        current_set.append(char)

    if current_set:
        sets.append(current_set)

    return sets

def generate_templates(patterns):
    """Generate templates from detected patterns."""
    templates = []
    for pattern in patterns:
        if len(pattern) >= 5:  # Example heuristic for a usable set
            templates.append(
                f"{pattern[0]}{'━' * 3}{pattern[1]}\n"
                f"{pattern[2]}   {pattern[3]}\n"
                f"{pattern[4]}{'━' * 3}{pattern[5] if len(pattern) > 5 else pattern[4]}"
            )
    return templates

def search_fonts(directory):
    """Search for font files in a directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.ttf', '.otf'))]

def analyze_font(font_path, characters):
    """Analyze a single font for TUI compatibility."""
    try:
        font = ImageFont.truetype(font_path, size=40)
    except IOError:
        print(f"Error: Unable to load font from {font_path}")
        return None

    horizontal_chars, vertical_chars, corner_chars, connectors = catalog_characters(characters, font)

    print(f"Analyzing {os.path.basename(font_path)}...")
    print("\nHorizontal Line Characters:")
    print("".join(horizontal_chars))
    print("\nVertical Line Characters:")
    print("".join(vertical_chars))
    print("\nCorner Characters:")
    print("".join(corner_chars))
    print("\nConnectors:")
    print("".join(connectors))

    # Detect patterns and generate templates
    patterns = detect_patterns(horizontal_chars + vertical_chars + corner_chars + connectors)
    templates = generate_templates(patterns)

    print("\nGenerated Templates:")
    for template in templates:
        print(template)
        print()

    return {
        "font": os.path.basename(font_path),
        "horizontal": horizontal_chars,
        "vertical": vertical_chars,
        "corners": corner_chars,
        "connectors": connectors,
        "templates": templates,
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze fonts for TUI compatibility.")
    parser.add_argument("--font-dir", type=str, help="Directory to search for font files.", required=True)
    args = parser.parse_args()

    # Define characters to analyze (box-drawing, geometric shapes, printable ASCII)
    characters = (
        [chr(i) for i in range(0x2500, 0x2580)] +  # Box Drawing
        [chr(i) for i in range(0x25A0, 0x25FF)] +  # Geometric Shapes
        list(string.printable)
    )

    # Search for fonts and analyze each
    font_files = search_fonts(args.font_dir)
    results = []

    for font_path in font_files:
        result = analyze_font(font_path, characters)
        if result:
            results.append(result)

    print("\nSummary of Fonts Analyzed:")
    for result in results:
        print(f"Font: {result['font']}")
        print(f"Templates:\n{''.join(result['templates'])}")

if __name__ == "__main__":
    main()

