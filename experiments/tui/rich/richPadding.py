#!/usr/bin/env python3
#
# richPadding.py

from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel

# Initialize Console
console = Console()

# Example 1: Default Padding with Uniform Value
console.rule("Default Padding with Uniform Value")
default_padding = Padding("Hello, world!", pad=2, style="on blue")
console.print(default_padding)

# Example 2: Padding with Different Values (CSS Style)
console.rule("Padding with CSS Style")
css_padding = Padding(
    "Rich padding example with CSS-style padding.",
    pad=(2, 4, 1, 3),  # Top, Right, Bottom, Left
    style="on green",
)
console.print(css_padding)

# Example 3: Padding with Two Values (CSS Shortcut)
console.rule("Padding with Two Values")
shortcut_padding = Padding(
    "This is a shortcut padding example.",
    pad=(2, 4),  # Top-Bottom, Left-Right
    style="on magenta",
)
console.print(shortcut_padding)

# Example 4: Padding with Expand Set to False
console.rule("Padding with Expand=False")
non_expanding_padding = Padding(
    "This padding won't expand to fit the width.",
    pad=(1, 2),
    style="on red",
    expand=False,
)
console.print(non_expanding_padding)

# Example 5: Nested Padding
console.rule("Nested Padding")
nested_padding = Padding(
    Padding("Nested padding example.", pad=2, style="on yellow"),
    pad=(1, 3),
    style="on cyan",
)
console.print(nested_padding)

# Example 6: Using Padding with Renderables
console.rule("Padding Around a Renderable")
renderable_padding = Padding(
    Panel("This is a panel wrapped with padding."),
    pad=(2, 4),
    style="on blue",
)
console.print(renderable_padding)

# Example 7: Indentation
console.rule("Indentation Example")
indented_text = Padding.indent("This text is indented by 4 spaces.", level=4)
console.print(indented_text)

# Example 8: Indentation for Renderables
console.rule("Indented Renderable Example")
indented_renderable = Padding.indent(
    Panel("This is a panel with an indentation of 6."),
    level=6,
)
console.print(indented_renderable)

# Example 9: Padding Unpack Demonstration
console.rule("Padding Unpack Example")
pad_values = Padding.unpack((5, 10, 2, 4))  # Top, Right, Bottom, Left
console.print(f"Unpacked Padding Values: {pad_values}")

# Example 10: Mixed Padding and Style
console.rule("Mixed Padding and Style Example")
mixed_padding = Padding(
    "This text has both padding and a background style.",
    pad=(3, 5, 2, 4),
    style="on purple bold",
)
console.print(mixed_padding)

# Summary of Padding Features
console.rule("Summary of Padding Features")
console.print(
    Panel(
        """
1. Default Padding: Uniform padding around content.
2. CSS-Style Padding: Specify padding for top, right, bottom, and left.
3. Two-Value Padding: Shortcut for (Top-Bottom, Left-Right).
4. Expand=False: Prevents padding from stretching to full width.
5. Nested Padding: Padding within padding.
6. Padding Around Renderables: Apply padding to panels or other renderables.
7. Indentation: Use `indent` for quick text or renderable indentation.
8. Padding Unpack: Unpack CSS-style padding into individual values.
9. Mixed Padding and Style: Combine padding with styles for emphasis.
""",
        title="Summary of Padding Features",
        style="on white",
    )
)

