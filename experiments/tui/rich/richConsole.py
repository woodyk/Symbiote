#!/usr/bin/env python3
#
# richConsole.py

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
from rich.theme import Theme
from rich.status import Status
from time import sleep
from io import StringIO

# Initialize Console
console = Console()

# Example 1: Basic Printing
console.rule("Example 1: Basic Printing")
console.print("Hello, [bold magenta]World![/bold magenta]")

# Example 2: Styled Text
console.rule("Example 2: Styled Text")
console.print("This is [bold red]important[/bold red]!", style="italic underline")

# Example 3: Pretty Printing Python Objects
console.rule("Example 3: Pretty Printing Python Objects")
console.print([1, 2, 3, {"key": "value"}])

# Example 4: Justify and Overflow
console.rule("Example 4: Justify and Overflow")
console.print("Center Align", justify="center")
console.print("Overflow Example", overflow="ellipsis", width=10)

# Example 5: Logging
console.rule("Example 5: Logging")
console.log("Starting the application...")

# Example 6: Logging with Locals
console.rule("Example 6: Logging with Locals")


def test_function():
    x = 10
    y = 20
    console.log("Logging locals", log_locals=True)


test_function()

# Example 7: Drawing Rules
console.rule("Example 7: Drawing Rules")
console.rule("[bold red]Section Title[/bold red]", style="dim blue", align="left")

# Example 8: Status Indicators
console.rule("Example 8: Status Indicators")
with console.status("Working on it...", spinner="dots"):
    sleep(2)

# Example 9: Pretty Printing JSON
console.rule("Example 9: Pretty Printing JSON")
console.print_json('[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]')

# Example 10: Input Prompt
console.rule("Example 10: Input Prompt")
name = console.input("Enter your [bold green]name[/]: ")
console.print(f"Hello, [italic]{name}[/italic]!")

# Example 11: Capturing Output
console.rule("Example 11: Capturing Output")
with console.capture() as capture:
    console.print("[bold red]Captured output[/bold red]")
output = capture.get()
console.print(f"Captured Output: {output}")

# Example 12: Alternate Screens
with console.screen(style=""):
    console.rule("Example 12: Alternate Screens")
    console.print(Panel("Full-screen message!"))
    sleep(2)

# Example 13: Paging
with console.pager():
    console.rule("Example 13: Paging")
    console.print("A very long text...\n" * 100)

# Example 14: Handling Dumb Terminals
console.rule("Example 14: Handling Dumb Terminals")
if console.is_dumb_terminal:
    console.print("Dumb terminal detected. Limited functionality.")
else:
    console.print("Terminal supports rich features!")

# Example 15: Exporting Output
console.rule("Example 15: Exporting Output")
output = """
recording_console = Console(record=True)
recording_console.print("Recording this message.")
recording_console.save_text("output.txt")
recording_console.save_html("output.html")
recording_console.save_svg("output.svg")
"""
console.print(Syntax(output, "python"))

# Example 16: Using Themes
console.rule("Example 16: Using Themes")
custom_theme = Theme({"info": "dim cyan", "warning": "bold yellow", "error": "bold red"})
themed_console = Console(theme=custom_theme)
themed_console.print("[info]This is an info message.")
themed_console.print("[warning]This is a warning.")
themed_console.print("[error]This is an error!")

# Example 17: JSON Rendering in Logs
console.rule("Example 17: JSON Rendering in Logs")
console.log(JSON('[{"key": "value"}]'))

# Example 18: Input Password
console.rule("Example 18: Input Password")
password = console.input("Enter your [bold red]password[/]: ", password=True)
console.print(f"Password captured: [not displayed]")

# Example 19: Status with Spinner
console.rule("Example 19: Status with Spinner")
with console.status("Loading...", spinner="dots", spinner_style="bold green"):
    sleep(2)

# Example 20: Error Handling
console.rule("Example 20: Error Handling")
try:
    1 / 0
except ZeroDivisionError:
    console.print_exception(show_locals=True)

