#!/usr/bin/env python3
#
# richSyntax.py

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from pygments.styles import get_all_styles

# Initialize Console
console = Console()

# Sample code to demonstrate syntax highlighting
sample_code = """
def greet(name: str) -> str:
    \"\"\"Greet a person by name.\"\"\"
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
"""

# Example 1: Basic Syntax Highlighting
console.rule("Example 1: Basic Syntax Highlighting")
syntax = Syntax(sample_code, "python", theme="monokai", line_numbers=True)
console.print(syntax)

# Example 2: Highlight Specific Lines
console.rule("Example 2: Highlight Specific Lines")
highlighted_syntax = Syntax(
    sample_code,
    "python",
    theme="monokai",
    line_numbers=True,
    highlight_lines={2, 5},
)
console.print(highlighted_syntax)

# Example 3: Custom Background Color
console.rule("Example 3: Custom Background Color")
custom_bg_syntax = Syntax(
    sample_code, "python", theme="monokai", background_color="rgb(240,240,240)"
)
console.print(custom_bg_syntax)

# Example 4: Displaying All Pygments Themes
console.rule("Example 4: Pygments Themes with Examples")

theme_table = Table(
    title="Pygments Themes",
    header_style="bold magenta",
    expand=True,
    show_lines=True,
)
theme_table.add_column("Theme Name", style="bold cyan", justify="center")
theme_table.add_column("Sample Syntax", style="bold green", justify="left")

# Loop through all available Pygments themes
for theme_name in sorted(get_all_styles()):
    syntax = Syntax(sample_code, "python", theme=theme_name, line_numbers=False)
    theme_table.add_row(theme_name, syntax)

# Center the table by wrapping it in a "centered" console print
console.print(theme_table)

# Example 5: Syntax from a File
console.rule("Example 5: Syntax Highlighting from a File")
# Create a file for demonstration
with open("sample.py", "w") as file:
    file.write(sample_code)

file_syntax = Syntax.from_path("sample.py", theme="dracula", line_numbers=True)
console.print(file_syntax)

# Example 6: Using Word Wrap and Padding
console.rule("Example 6: Word Wrap and Padding")
wrapped_syntax = Syntax(
    sample_code,
    "python",
    theme="emacs",
    line_numbers=True,
    word_wrap=True,
    padding=(1, 2),
)
console.print(wrapped_syntax)

# Cleanup the created file
import os
os.remove("sample.py")

