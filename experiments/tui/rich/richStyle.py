#!/usr/bin/env python3
#
# richStyle.py

from rich.console import Console
from rich.style import Style
from rich.theme import Theme

# Initialize Console
console = Console()

# Example 1: Basic Style Usage
console.rule("Example 1: Basic Style Usage")
console.print("Hello, World!", style="bold magenta")
console.print("Danger!", style="red on yellow")
console.print("Custom color", style="#ff5733")

# Example 2: Advanced Color Definitions
console.rule("Example 2: Advanced Color Definitions")
console.print("Using color numbers", style="color(82)")
console.print("Using RGB colors", style="rgb(128,0,128)")
console.print("Background color only", style="on rgb(255,215,0)")

# Example 3: Combining Attributes
console.rule("Example 3: Combining Attributes")
console.print("Bold, Underlined, and Italic", style="bold underline italic cyan")
console.print("Reverse colors", style="reverse red on black")
console.print("Strike-through text", style="strike magenta")

# Example 4: Hyperlink
console.rule("Example 4: Hyperlink")
console.print("Visit [link=https://www.example.com]Example Site[/link]", style="bold green")

# Example 5: Using the `Style` Class
console.rule("Example 5: Using the Style Class")
custom_style = Style(color="red", bgcolor="yellow", bold=True, italic=True)
console.print("Custom Style Example", style=custom_style)

# Example 6: Combining Styles
console.rule("Example 6: Combining Styles")
base_style = Style.parse("cyan")
combined_style = base_style + Style(underline=True)
console.print("Combined Style Example", style=combined_style)

# Example 7: Parsing Style Definitions
console.rule("Example 7: Parsing Style Definitions")
parsed_style = Style.parse("italic magenta on yellow")
console.print("Parsed Style Example", style=parsed_style)

# Example 8: Using a Theme
console.rule("Example 8: Using a Theme")
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})
themed_console = Console(theme=custom_theme)
themed_console.print("This is information", style="info")
themed_console.print("[warning]The pod bay doors are locked[/warning]")
themed_console.print("Something terrible happened!", style="danger")

# Example 9: Customizing Defaults
console.rule("Example 9: Customizing Defaults")
default_theme_console = Console(theme=Theme({"repr.number": "bold green blink"}))
default_theme_console.print("The total is 128")

# Example 10: Using `StyleStack`
console.rule("Example 10: Using StyleStack")
from rich.style import StyleStack

style_stack = StyleStack(default_style=Style(color="white"))
style_stack.push(Style(color="blue", bold=True))
console.print("Current Style", style=style_stack.current)
style_stack.pop()
console.print("After Pop", style=style_stack.current)

# Example 11: External Config File for Themes
console.rule("Example 11: External Config File")
output = """
    theme_from_file = Theme.read("theme.cfg")  # Ensure theme.cfg exists
    file_theme_console = Console(theme=theme_from_file)
    file_theme_console.print("Loaded theme from file.", style="info")
"""
console.print(output)

# Example 12: Advanced Styles
console.rule("Example 12: Advanced Styles")
console.print("Double Underline", style="underline2 yellow")
console.print("Framed Text", style="frame cyan")
console.print("Encircled Text", style="encircle red")
console.print("Overlined Text", style="overline green")

