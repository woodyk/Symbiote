#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: splash.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-30 19:07:45
# Modified: 2025-03-23 14:02:27

import sys
from pyfiglet import Figlet

# Predefined gradient themes
gradients = {
    "sunset": ("#FF4500", "#FFD700"),
    "ocean": ("#1E90FF", "#00CED1"),
    "forest": ("#2E8B57", "#ADFF2F"),
    "fire": ("#FF0000", "#FFA500"),
    "rainbow": ("#9400D3", "#FF0000"),  # Purple to Red
    "cotton_candy": ("#FFB6C1", "#ADD8E6"),
    "lava": ("#800000", "#FF4500"),
    "electric": ("#00FFFF", "#7B68EE"),
    "peach": ("#FFDAB9", "#FF6347"),
    "aurora": ("#4B0082", "#00FF00"),
    "twilight": ("#8A2BE2", "#FF69B4"),  # Purple to Pink
    "neon_city": ("#00FF00", "#FF00FF"),  # Green to Magenta
    "desert": ("#EDC9AF", "#FF4500"),  # Sandy Beige to Orange
    "iceberg": ("#00FFFF", "#FFFFFF"),  # Cyan to White
    "galaxy": ("#000080", "#8A2BE2"),  # Deep Blue to Purple
    "rose_garden": ("#FF007F", "#FF69B4"),  # Dark Pink to Soft Pink
    "sunrise": ("#FFD700", "#FF4500"),  # Gold to Orange
    "serenity": ("#00CED1", "#4682B4"),  # Teal to Steel Blue
    "midnight": ("#191970", "#000000"),  # Navy to Black
    "golden_hour": ("#FFD700", "#FFA500"),  # Gold to Deep Orange
}

def hex_to_rgb(hex_color):
    """
    Convert a hex color to an RGB tuple.
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_ansi(r, g, b):
    """
    Convert RGB to ANSI 256 color.
    """
    return 16 + 36 * (r // 51) + 6 * (g // 51) + (b // 51)

def gradient_text(text, start_color, end_color):
    """
    Apply a gradient to the text using ANSI escape codes.
    """
    lines = text.splitlines()
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    gradient_output = []

    for i, line in enumerate(lines):
        if not line.strip():  # Skip empty lines
            gradient_output.append("")
            continue

        factor = i / max(len(lines) - 1, 1)
        current_rgb = (
            int(start_rgb[0] + factor * (end_rgb[0] - start_rgb[0])),
            int(start_rgb[1] + factor * (end_rgb[1] - start_rgb[1])),
            int(start_rgb[2] + factor * (end_rgb[2] - start_rgb[2])),
        )
        color_code = rgb_to_ansi(*current_rgb)
        colored_line = f"\033[38;5;{color_code}m{line}\033[0m"
        gradient_output.append(colored_line)

    return "\n".join(gradient_output)

def render_gradient_figlet(text, theme="rainbow", font="slant"):
    """
    Render text with a figlet font and apply a gradient theme using ANSI colors.

    Args:
        text (str): The text to render.
        theme (str): The gradient theme name.
        font (str): The figlet font to use.
    """
    if theme not in gradients:
        print(f"Error: Theme '{theme}' not found.", file=sys.stderr)
        return

    # Generate figlet text
    figlet = Figlet(font=font)
    rendered_text = figlet.renderText(text)

    # Apply gradient
    start_color, end_color = gradients[theme]
    gradient_output = gradient_text(rendered_text, start_color, end_color)

    print(gradient_output)

if __name__ == "__main__":
    def list_figlet_fonts(sample_text="Symbiote", theme="rainbow"):
        """
        Print an example of every Figlet font with a gradient applied.

        Args:
            sample_text (str): The text to display in each font.
            theme (str): The gradient theme to apply.
        """
        figlet = Figlet()
        fonts = figlet.getFonts()
        print(f"Rendering {len(fonts)} fonts with the '{theme}' theme...\n")

        for font in fonts:
            print(f"Font: {font}")
            figlet.setFont(font=font)
            rendered_text = figlet.renderText(sample_text)
            gradient_output = gradient_text(rendered_text, *gradients[theme])
            print(gradient_output)
            print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    # Display an example of every Figlet font
    #list_figlet_fonts(sample_text="Symbiote", theme="galaxy")

    figlet = Figlet()
    fonts = figlet.getFonts()
    for theme in gradients:
        rendered_text = figlet.renderText("symbiote")
        gradient_output = gradient_text(rendered_text, *gradients[theme])
        print(f"--- {theme} ---")
        print(gradient_output)


