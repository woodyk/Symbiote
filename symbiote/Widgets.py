#!/usr/bin/env python3
#
# Widgets.py

import os
from pathlib import Path
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Frame
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich import box
from rich.console import Console
from rich.text import Text
from rich.align import Align
from rich.box import SQUARE, DOUBLE
from rich.table import Table
from rich.theme import Theme

class TerminalWidgets:
    def __init__(self):
        # Define a custom theme for consistent styling
        theme = Theme({
            "directory": "bold yellow",
            "file": "white",
            "selected": "bold green",
            "footer": "dim white",
            "key_column": "cyan",
            "value_column": "magenta",
            "dialog_title": "bold white",
            "dialog_text": "white",
            "dialog_border": "bold white",
            "prompt_text": "bold cyan"
        })
        # Create a console with the custom theme for rich output
        self.console = Console(theme=theme)
        self.kb = KeyBindings()

    
    def show_popup_dialog(self, message: str, title: str = "Message"):
        """Display a centered popup dialog with a title and message using rich with a box border."""
        # Clear the screen
        os.system('clear' if os.name == 'posix' else 'cls')

        # Create a table styled as a dialog box with a border using the `box` attribute
        table = Table.grid(expand=False, padding=(1, 2))  # Centered dialog with padding
        table.add_column(justify="center")
        
        # Add title and message with themed styles
        table.add_row(Text(title, style="dialog_title"))
        table.add_row(Align.center(Text(message, style="dialog_text")))
        
        # Apply box style and border from the theme
        table.box = SQUARE
        table.border_style = "dialog_border"

        # Print the dialog centered on the screen and wait for Enter to continue
        self.console.print(Align.center(table), justify="center")
        self.console.print("\nPress [prompt_text]Enter[/prompt_text] to continue...", justify="center")
        input()  # Wait for the user to press Enter to continue

        # Clear the screen again after the dialog is dismissed
        os.system('clear' if os.name == 'posix' else 'cls')

    def prompt_file_browser(self):
        """Terminal-based file browser using prompt_toolkit to navigate and select files."""
        current_path = Path.home()
        files = []
        selected_index = 0
        scroll_offset = 0
        show_hidden = False

        terminal_height = int(os.get_terminal_size().lines / 2)
        max_display_lines = terminal_height - 2

        def update_file_list():
            nonlocal files, selected_index, scroll_offset
            all_files = [".."] + sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            files = [f for f in all_files if isinstance(f, str) or show_hidden or not f.name.startswith('.')]
            selected_index = 0
            scroll_offset = 0

        def get_display_text():
            text = []
            visible_files = files[scroll_offset:scroll_offset + max_display_lines]
            for i, f in enumerate(visible_files):
                real_index = scroll_offset + i
                prefix = "> " if real_index == selected_index else "  "
                color = "green" if real_index == selected_index else ("yellow" if isinstance(f, Path) and f.is_dir() else "white")
                display_name = f if isinstance(f, str) else f.name
                display_name += "/" if isinstance(f, Path) and f.is_dir() else ""
                line = f"{prefix}{display_name}"
                text.append((color, line))
                text.append(('', '\n'))
            return text

        update_file_list()

        @self.kb.add("up")
        def move_up(event):
            nonlocal selected_index, scroll_offset
            selected_index = (selected_index - 1) % len(files)
            if selected_index < scroll_offset:
                scroll_offset = max(0, scroll_offset - 1)

        @self.kb.add("down")
        def move_down(event):
            nonlocal selected_index, scroll_offset
            selected_index = (selected_index + 1) % len(files)
            if selected_index >= scroll_offset + max_display_lines:
                scroll_offset = min(len(files) - max_display_lines, scroll_offset + 1)

        @self.kb.add("enter")
        def enter_directory(event):
            nonlocal current_path
            selected_file = files[selected_index]
            if selected_file == "..":
                current_path = current_path.parent
                update_file_list()
            elif isinstance(selected_file, Path) and selected_file.is_dir():
                current_path = selected_file
                update_file_list()
            elif isinstance(selected_file, Path) and selected_file.is_file():
                event.app.exit(result=str(selected_file))

        @self.kb.add("escape")
        def cancel_selection(event):
            event.app.exit(result=None)

        @self.kb.add("c-h")
        def toggle_hidden(event):
            nonlocal show_hidden
            show_hidden = not show_hidden
            update_file_list()

        file_list_window = Window(content=FormattedTextControl(get_display_text), wrap_lines=False, height=max_display_lines)
        footer_window = Window(content=FormattedTextControl(lambda: "Press Ctrl-H to show/hide hidden files. Escape to exit."), height=1, style="gray")
        layout = Layout(HSplit([
            Frame(Window(FormattedTextControl(lambda: f"Current Directory: {current_path}"), height=1)),
            file_list_window,
            footer_window
        ]))

        app = Application(layout=layout, key_bindings=self.kb, full_screen=True, refresh_interval=0.1)
        return app.run()

    def prompt_selection_menu(self, options):
        """Selection menu to pick an item from a list or dictionary."""
        if isinstance(options, dict):
            display_options = list(options.keys())
            values = list(options.values())
        elif isinstance(options, list):
            display_options = options
            values = options
        else:
            raise ValueError("Options must be a list or a dictionary.")

        selected_index = 0
        max_display_lines = min(len(display_options), int(os.get_terminal_size().lines / 2) - 2)

        def get_display_text():
            text = []
            visible_options = display_options[:max_display_lines]
            for i, option in enumerate(visible_options):
                prefix = "> " if i == selected_index else "  "
                color = "green" if i == selected_index else "white"
                line = f"{prefix}{option}"
                text.append((color, line))
                text.append(('', '\n'))
            return text

        @self.kb.add("up")
        def move_up(event):
            nonlocal selected_index
            selected_index = (selected_index - 1) % len(display_options)

        @self.kb.add("down")
        def move_down(event):
            nonlocal selected_index
            selected_index = (selected_index + 1) % len(display_options)

        @self.kb.add("enter")
        def select_option(event):
            event.app.exit(result=values[selected_index])

        @self.kb.add("escape")
        def cancel_selection(event):
            event.app.exit(result=None)

        selection_window = Window(content=FormattedTextControl(get_display_text), wrap_lines=False, height=max_display_lines)
        footer_window = Window(content=FormattedTextControl(lambda: "Use Up/Down to navigate, Enter to select, Escape to cancel."), height=1, style="gray")
        layout = Layout(HSplit([
            Frame(Window(FormattedTextControl(lambda: "Select an option:"), height=1)),
            selection_window,
            footer_window
        ]))

        app = Application(layout=layout, key_bindings=self.kb, full_screen=True, refresh_interval=0.1)
        return app.run()

    def render_table(self, data):
        """Render a table with two columns: key and value using rich's Table widget, filling the terminal width with borders."""
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary.")

        # Get the full width of the terminal
        terminal_width = self.console.size.width

        # Create a rich table with full width and borders
        table = Table(title="Data Table", expand=True, box=box.SQUARE)
        table.add_column("Key", style="key_column", no_wrap=True, width=int(terminal_width * 0.3))
        table.add_column("Value", style="value_column", width=int(terminal_width * 0.7))

        # Add rows for each key-value pair in the dictionary
        for key, value in data.items():
            table.add_row(str(key), str(value))

        # Render the table to fill the terminal width
        self.console.print(table)

if __name__ == "__main__":
    # Instantiate the widget class
    widgets = TerminalWidgets()

    # Test the file browser
    print("Testing file browser...")
    selected_file = widgets.prompt_file_browser()
    if selected_file:
        print(f"Selected file: {selected_file}")
    else:
        print("File browser canceled.")

    # Test the selection menu with a list
    print("\nTesting selection menu with a list...")
    list_options = ["Option 1", "Option 2", "Option 3"]
    selected_option = widgets.prompt_selection_menu(list_options)
    if selected_option:
        print(f"Selected option: {selected_option}")
    else:
        print("Selection menu canceled.")

    # Test the selection menu with a dictionary
    print("\nTesting selection menu with a dictionary...")
    dict_options = {"Option A": "Value A", "Option B": "Value B", "Option C": "Value C"}
    selected_option = widgets.prompt_selection_menu(dict_options)
    if selected_option:
        print(f"Selected option: {selected_option}")
    else:
        print("Selection menu canceled.")

    # Test the popup dialog
    print("\nTesting popup dialog...")
    widgets.show_popup_dialog("This is a test popup message.")

    # Test the table renderer
    print("\nTesting table renderer...")
    data = {
        "Name": "Alice",
        "Age": 30,
        "Occupation": "Engineer",
        "Location": "New York"
    }
    widgets.render_table(data)


