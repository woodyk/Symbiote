#!/usr/bin/env python3
#
# richTraceback.py

from rich.console import Console
from rich.traceback import install, Traceback

# Initialize Console
console = Console()

# Install Rich as the default traceback handler
console.rule("Installing Rich Traceback Handler")
install(show_locals=True, suppress=["os", "sys"])

# Example 1: Basic Exception Handling
console.rule("Example 1: Basic Exception Handling")
try:
    1 / 0
except ZeroDivisionError:
    console.print_exception(show_locals=False)

# Example 2: Show Locals
console.rule("Example 2: Show Locals")
def sample_function():
    a = 42
    b = "Rich Traceback"
    c = [1, 2, 3]
    raise ValueError("Something went wrong!")

try:
    sample_function()
except Exception:
    console.print_exception(show_locals=True)

# Example 3: Custom Traceback Appearance
console.rule("Example 3: Custom Traceback Appearance")
try:
    "rich" + 5
except Exception:
    traceback = Traceback(
        width=80,
        extra_lines=2,
        theme="monokai",
        word_wrap=True,
        show_locals=True,
        indent_guides=False,
    )
    console.print(traceback)

# Example 4: Suppress Framework Code
console.rule("Example 4: Suppress Framework Code")
import click

@click.command()
def cli_function():
    def nested_function():
        raise RuntimeError("Error inside a CLI function!")

    nested_function()

try:
    cli_function()
except Exception:
    console.print_exception(suppress=[click])

# Example 5: Handling Recursive Errors
console.rule("Example 5: Handling Recursive Errors")

def recursive_function(n):
    if n == 0:
        raise RecursionError("Recursion limit reached!")
    return recursive_function(n - 1)

try:
    recursive_function(50)
except Exception:
    console.print_exception(max_frames=10)

# Example 6: Manual Traceback Rendering
console.rule("Example 6: Manual Traceback Rendering")
try:
    "text" + 10
except Exception as exc:
    tb = Traceback.from_exception(
        exc_type=type(exc),
        exc_value=exc,
        traceback=exc.__traceback__,
        width=80,
        show_locals=True,
        max_frames=5,
    )
    console.print(tb)

# Example 7: Customizing the Default Handler
console.rule("Example 7: Customizing the Default Handler")

def buggy_function():
    d = {"key1": "value1", "key2": "value2"}
    return d["missing_key"]

try:
    buggy_function()
except Exception:
    console.print_exception(show_locals=True, locals_max_length=5, locals_max_string=20)

