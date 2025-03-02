#!/usr/bin/env python3
#
# sym_reference.py
"""
Module: sym_reference

This module serves as a collection of example templates for common use cases with various Python modules.
It provides concise, easy-to-follow examples to demonstrate best practices, common patterns, and effective
usage of Python libraries.

Each template is designed to be standalone and focuses on specific functionality or feature sets within
the respective module. The module aims to help developers quickly understand and implement these features
in their own projects.

Label templates::
---------

# ---------- Function Definitions ----------
# -------------------------------------------
# ===========================================
# *******************************************
# ======= Section: Data Processing ==========
# ===========================================
# Section: Import Statements
# ===========================================
# -------------------------------------------
# Section: Utility Functions
# -------------------------------------------
# *******************************************
# Section: Main Program Execution
# *******************************************
"""






# ===========================================
"""
Module: key_bindings_demo

This module demonstrates handling key bindings in a Python application using
a combination of the `Rich` library for visually appealing terminal interfaces
and `prompt_toolkit` for managing user input.

The primary functionality includes:
- Creating a full-screen, live-updating terminal interface with Rich.
- Displaying a help menu in a formatted table using Rich's `Table` and `Panel`.
- Managing key bindings (`q` and `Escape`) with `prompt_toolkit` to handle user interactions and exit the application gracefully.

Dependencies:
-------------
- `Rich`: For rendering tables, panels, and managing live updates.
- `prompt_toolkit`: For creating interactive CLI applications and handling keyboard events.

Functionality:
--------------
- Displays a formatted command help table with Rich.
- Captures user input using prompt_toolkit key bindings.
- Exits the application when `q` or `Escape` is pressed.

"""
from rich.table import Table
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

def display_help():
        """
    Display a full-screen help menu with key binding management.

    This function showcases the integration of Rich and prompt_toolkit
    to create a dynamic terminal interface. It renders a table of commands
    using Rich's Table and Panel, and utilizes prompt_toolkit's key bindings
    to capture user input for exiting the application.

    Features:
    ---------
    - Displays a table of commands with their descriptions.
    - Handles `q` and `Escape` key presses to exit gracefully.
    - Uses Rich's Live context manager for real-time updates.

    How it works:
    -------------
    - A Rich `Table` is created and displayed within a `Panel` that
      includes instructions for exiting.
    - prompt_toolkit's `KeyBindings` are used to bind the `q` and `Escape`
      keys to an exit function.
    - The application runs in full-screen mode and exits cleanly when
      a bound key is pressed.

    Notes:
    ------
    - This example is extendable to include additional key bindings or features.
    - Ensure all required libraries are installed before running.

    Usage:
    ------
    Run the script and observe the interactive help menu. Press `q` or `Escape`
    to exit.

    Example:
    --------
    >>> display_help()

    """
    console = Console()
    command_list = {
        "help": "Show this help message",
        "exit": "Exit the application",
        "run": "Run the main task",
        "status": "Show the current status",
        # Add more commands here
    }
    table = Table(show_header=True, expand=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for cmd, desc in sorted(command_list.items()):
        table.add_row(cmd, desc)

    # Panel with instructions
    panel = Panel(
        table,
        title="Command Help",
        subtitle="Press 'q' or 'Escape' to exit",
        border_style="green",
    )

    # Key bindings for prompt_toolkit
    kb = KeyBindings()

    @kb.add("q")  # Bind the 'q' key
    @kb.add(Keys.Escape)  # Bind the 'Escape' key
    def _(event):
        live.stop()
        event.app.exit(result=None)

    # Create a prompt_toolkit application with the key bindings
    app = Application(full_screen=False, key_bindings=kb)
    # app.is_running: bool
    # app.in_terminal: bool
    # app.run(): starts the app

    # Run the Live display in the application context
    with Live(panel, console=console, screen=True, refresh_per_second=10) as live:
        app.run()
        while not app.is_running:
            time.sleep(0.1)

if __name__ == "__main__":
    display_help()

"""
This script demonstrates a simple template for using `prompt_toolkit`'s `PromptSession` 
with custom key bindings, without relying on `Application` or other complex constructs. 
It integrates with the `rich` library for dynamic, styled terminal output and provides 
an interactive help system.

Functions:
----------
1. display_help():
    - Displays a list of commands in a styled table using the `rich` library.
    - Provides key bindings for navigation and exiting the help interface.
    - Uses `PromptSession` to capture user input, allowing dynamic updates.

Features:
---------
- Command Table:
    - Commands are displayed in a table with their descriptions, styled using `rich.Table`.
    - Encased in a `rich.Panel` for clear visualization, with a title and instructions.

- Key Bindings:
    - Custom key bindings allow users to exit (`q`, `Escape`, `Enter`, or `Space`) 
      or handle unspecified keys gracefully (`Keys.Any`).
    - Key bindings are defined using `prompt_toolkit.key_binding.KeyBindings`.

- Dynamic Console:
    - The `rich.live.Live` feature keeps the panel visible and refreshes dynamically 
      during user input.

- Example Usage:
    - Run the script to display the help interface and interact with it using the defined keys.

Customization:
--------------
- The `command_list` dictionary defines available commands and their descriptions.
  Add or modify commands as needed.
- Modify key bindings to match your application's needs.
- Adjust styling in `rich.Table` and `rich.Panel` for a customized appearance.

Usage:
------
Run the script to see the help interface in action. The `display_help` function can 
also be adapted or integrated into larger projects for an interactive terminal-based 
help system.

Dependencies:
-------------
- `rich`: For creating styled and dynamic terminal interfaces.
- `prompt_toolkit`: For handling key bindings and interactive prompts.

Note:
-----
This script is intended as a simple and extensible template. For complex applications, 
consider using `prompt_toolkit`'s `Application` for enhanced features.
"""
from rich.table import Table
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
import time

def display_help():
    console = Console()
    command_list = {
        "help": "Show this help message",
        "exit": "Exit the application",
        "run": "Run the main task",
        "status": "Show the current status",
        # Add more commands here
    }
    table = Table(show_header=True, expand=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for cmd, desc in sorted(command_list.items()):
        table.add_row(cmd, desc)

    # Panel with instructions
    panel = Panel(
        table,
        title="Command Help",
        subtitle="Press 'q' or 'Escape' to exit",
        height=console.height,
    )

    # Key bindings for prompt_toolkit
    help_keys = KeyBindings()

    @help_keys.add(" ")  # Bind the 'q' key
    @help_keys.add("q")  # Bind the 'q' key
    @help_keys.add("enter")
    @help_keys.add("escape")
    @help_keys.add(Keys.Any)  # Catch any key not explicitly handled
    def _(event):
        event.app.exit()

    help_session = PromptSession(key_bindings=help_keys)

    with Live(panel, console=console, screen=True, refresh_per_second=10) as live:
        user_input = help_session.prompt(key_bindings=help_keys)

if __name__ == "__main__":
    display_help()

# ===========================================
# Make a prompt_toolkit PromptSession stop blocking
# by sending input data to the prompt remotely

from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.key_binding import KeyBindings

# Send blocking prompt data to continue
def start_prompt_sesion():
    kb = KeyBindings()

    # create your key binding
    @kb.add("i")
    def _i_pressed(event):
        # Insert text into the current buffer
        session.app.current_buffer.insert_text(text)

        # Simulate the Enter key press
        session.app.current_buffer.validate_and_handle()

    # Configure the session with the
    # key board shortcut
    s = PromptSession(key_bindings=kb)

    # Start trigger the prompt.
    # the s.prompt is currently blocking
    user_input = s.prompt("prompt>", key_bindings=kb)

    # print something to test
    # if seen without hitting enter
    # at the prompt then prompt unblocked
    print(f"{OH SHIT}: {user_input}")


# using curses to manage keyboard input
import curses

def main(stdscr):  # stdscr is passed automatically by curses.wrapper
    # Clear the screen
    stdscr.clear()

    # Turn off cursor
    curses.curs_set(0)

    # Enable keypad input
    stdscr.keypad(True)

    # Instructions
    stdscr.addstr(0, 0, "Press 'q' to exit, arrow keys to move the cursor", curses.A_BOLD)

    # Initialize cursor position
    y, x = 1, 1

    while True:
        # Display current position
        stdscr.addstr(2, 0, f"Current position: y={y}, x={x}   ")

        # Refresh the screen
        stdscr.refresh()

        # Get user input
        key = stdscr.getch()

        # Handle key presses
        if key == ord('q'):  # Exit on 'q'
            break
        elif key == curses.KEY_UP:  # Move up
            y = max(1, y - 1)
        elif key == curses.KEY_DOWN:  # Move down
            y = min(curses.LINES - 1, y + 1)
        elif key == curses.KEY_LEFT:  # Move left
            x = max(0, x - 1)
        elif key == curses.KEY_RIGHT:  # Move right
            x = min(curses.COLS - 1, x + 1)

        # Clear and draw the cursor
        stdscr.clear()
        stdscr.addstr(0, 0, "Press 'q' to exit, arrow keys to move the cursor", curses.A_BOLD)
        stdscr.addstr(y, x, "X", curses.A_REVERSE)

# Run the curses application
curses.wrapper(main)


