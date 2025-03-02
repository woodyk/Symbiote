#!/usr/bin/env python3
#
# richGradientRule.py

from rich.console import Console
from rich.text import Text


def gradient_rule(
    text: str = "",
    *,
    start_color: str = "#FF0000",
    end_color: str = "#0000FF",
    char: str = "─",
    pattern: str = "default",
    style: str = "",
    width: int = None,
    font: str = None,
):
    """
    Display a horizontal rule with a gradient color effect.

    Args:
        text (str): Optional text to display in the middle of the rule.
        start_color (str): The starting color of the gradient.
        end_color (str): The ending color of the gradient.
        char (str): The character used to draw the rule.
        pattern (str): Predefined pattern for the rule.
        style (str): Additional style to apply to the rule.
        width (int): Width of the rule. Defaults to the terminal width.
        font (str): Custom font style (e.g., "bold", "italic") for the text.
    """
    console = Console()
    console_width = console.size.width if width is None else width

    # Predefined patterns
    patterns = {
        "default": "─",
        "block": "█",
        "double": "═",
        "wave": "≈",
        "dot": "•",
        "thick": "▇",
        "arrow": "→",
        "star": "✦",
        "diamond": "◆",
        "zigzag": "╱",
        "plus": "+",
    }
    char = patterns.get(pattern, char)

    # Gradient interpolation
    def interpolate_color(ratio):
        r1, g1, b1 = tuple(int(start_color[i : i + 2], 16) for i in (1, 3, 5))
        r2, g2, b2 = tuple(int(end_color[i : i + 2], 16) for i in (1, 3, 5))
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        return f"rgb({r},{g},{b})"

    # Construct the rule
    rule = Text()
    text_length = len(text) + 2 if text else 0  # Account for text and padding
    half_width = (console_width - text_length) // 2

    # Left half of the rule
    for i in range(half_width):
        ratio = i / console_width
        color = interpolate_color(ratio)
        rule.append(char, style=f"{style} {color}")

    # Add centered text if present, apply font style
    if text:
        styled_text = Text(f" {text} ", style=font if font else "")
        rule.append(styled_text)

    # Right half of the rule
    for i in range(half_width + text_length, console_width):
        ratio = i / console_width
        color = interpolate_color(ratio)
        rule.append(char, style=f"{style} {color}")

    console.print(rule)


# Corrected Predefined Rules
def display_predefined_rules():
    gradient_rule(
        "Gradient Rule",
        start_color="#FF0000",
        end_color="#00FF00",
        pattern="wave",
        font="bold",
    )
    gradient_rule(
        "Block Style",
        start_color="#0000FF",
        end_color="#FF00FF",
        pattern="block",
        font="italic",
    )
    gradient_rule(
        "Thick Divider",
        start_color="#FFAA00",
        end_color="#5500FF",
        pattern="thick",
    )
    gradient_rule(
        "Dot Pattern",
        start_color="#00FFFF",
        end_color="#FF00FF",
        pattern="dot",
    )
    gradient_rule(
        "Arrow Divider",
        start_color="#FF0000",
        end_color="#FFFF00",
        pattern="arrow",
    )
    gradient_rule(
        "Starry Rule",
        start_color="#00FF00",
        end_color="#00FFFF",
        pattern="star",
    )
    gradient_rule(
        "Diamond Rule",
        start_color="#FF00FF",
        end_color="#0000FF",
        pattern="diamond",
    )
    gradient_rule(
        "Zigzag Divider",
        start_color="#5500FF",
        end_color="#AAFF00",
        pattern="zigzag",
    )
    gradient_rule(
        "Plus Line",
        start_color="#FF0000",
        end_color="#00FF00",
        pattern="plus",
    )
    gradient_rule(
        start_color="#000000",
        end_color="#FFFFFF",
        pattern="double",
    )


# Run Examples
display_predefined_rules()

