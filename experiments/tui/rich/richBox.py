#!/usr/bin/env python3
#
# richBox.py

from rich.console import Console
from rich import box
from rich.panel import Panel
from rich.text import Text

console = Console()

def render_box_examples():
    """
    Render an example of all available box styles in rich.box.
    """
    console.rule("Rich Box Style Demonstration")

    # Get all box types from the box module
    border_styles = [(name, getattr(box, name)) for name in dir(box) if name.isupper()]

    # Render an example for each box style
    for name, style in border_styles:
        title = f"Box Style: {name}"
        example_text = Text(f"This is an example of the {name} box style.", justify="center")
        panel = Panel(example_text, box=style, title=title, title_align="left")

        try:
            console.print(panel)
        except:
            pass

        console.line()

    console.rule("End of Box Style Demonstration")

if __name__ == "__main__":
    render_box_examples()

