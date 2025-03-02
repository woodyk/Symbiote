#!/usr/bin/env python3
#
# richPanel.py

from rich.console import Console
from rich.panel import Panel
from rich import box

# Initialize Console
console = Console()

# Example 1: Basic Panel
console.rule("Basic Panel Example")
basic_panel = Panel("Hello, World!")
console.print(basic_panel)

# Example 2: Panel with Title
console.rule("Panel with Title Example")
panel_with_title = Panel("This is a panel with a title.", title="Title")
console.print(panel_with_title)

# Example 3: Panel with Subtitle
console.rule("Panel with Subtitle Example")
panel_with_subtitle = Panel("This panel has a subtitle.", subtitle="Subtitle")
console.print(panel_with_subtitle)

# Example 4: Title and Subtitle with Alignment
console.rule("Title and Subtitle Alignment Example")
aligned_panel = Panel(
    "Title and subtitle can be aligned.",
    title="Aligned Title",
    subtitle="Aligned Subtitle",
    title_align="left",
    subtitle_align="right",
)
console.print(aligned_panel)

# Example 5: Custom Box Style
console.rule("Custom Box Style Example")
custom_box_panel = Panel(
    "Custom box style using DOUBLE box.",
    box=box.DOUBLE,
    title="Custom Box",
)
console.print(custom_box_panel)

# Example 6: Safe Box for Compatibility
console.rule("Safe Box Example")
safe_box_panel = Panel(
    "Safe box characters for compatibility with older terminals.",
    box=box.SQUARE,
    safe_box=True,
    title="Safe Box",
)
console.print(safe_box_panel)

# Example 7: Panel with Padding
console.rule("Panel with Padding Example")
padded_panel = Panel("This panel has padding.", padding=(2, 4), title="Padding")
console.print(padded_panel)

# Example 8: Panel with Style
console.rule("Panel with Style Example")
styled_panel = Panel(
    "This panel has a custom style and border style.",
    style="on blue",
    border_style="bold magenta",
    title="Styled Panel",
)
console.print(styled_panel)

# Example 9: Panel with Width and Height
console.rule("Panel with Custom Dimensions Example")
dimensioned_panel = Panel(
    "This panel has a fixed width and height.",
    width=40,
    height=10,
    title="Custom Dimensions",
)
console.print(dimensioned_panel)

# Example 10: Panel with Highlighting
console.rule("Panel with Highlighting Example")
highlight_panel = Panel(
    "[bold magenta]This panel automatically highlights[/bold magenta] content.",
    highlight=True,
    title="Highlighting",
)
console.print(highlight_panel)

# Example 11: Using the `fit` Method
console.rule("Panel Using fit() Method Example")
fit_panel = Panel.fit(
    "This panel uses the fit method to automatically size.",
    box=box.ROUNDED,
    title="Fit Panel",
)
console.print(fit_panel)

# Example 12: Nested Panels
console.rule("Nested Panels Example")
nested_panel = Panel(
    Panel(
        "Inner panel with its own border.",
        title="Inner Panel",
        box=box.SQUARE,
    ),
    title="Outer Panel",
    box=box.DOUBLE,
)
console.print(nested_panel)

# Summary of Panel Features
console.rule("Summary of Panel Features")
console.print(
    Panel(
        """
1. Basic Panel: Simple panel with default settings.
2. Title: Add a title to the panel.
3. Subtitle: Add a subtitle to the panel.
4. Title and Subtitle Alignment: Align titles and subtitles to left, center, or right.
5. Custom Box Style: Use different box styles (e.g., ROUNDED, DOUBLE, SQUARE).
6. Safe Box: Ensures compatibility with older terminals.
7. Padding: Add spacing inside the panel content.
8. Style: Apply custom styles to the panel and its border.
9. Width and Height: Specify fixed dimensions for the panel.
10. Highlighting: Enable automatic syntax highlighting in the panel.
11. Fit Method: Automatically size the panel to fit its content.
12. Nested Panels: Panels within panels for complex layouts.
""",
        title="Summary",
        style="cyan",
    )
)

