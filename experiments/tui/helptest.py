#!/usr/bin/env python3
#
# helptest.py

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings


class CommandNavigator:
    def __init__(self, command_list):
        self.console = Console()
        self.command_list = sorted(command_list.items())
        self.commands_per_page = 10  # Number of rows to display per page
        self.current_page = 0

    def create_table(self):
        """Create a table for the current page of commands."""
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Command 1", style="cyan", no_wrap=True)
        table.add_column("Description 1", style="white")
        table.add_column("Command 2", style="cyan", no_wrap=True)
        table.add_column("Description 2", style="white")

        # Split the command list for pagination
        start_index = self.current_page * self.commands_per_page
        end_index = start_index + self.commands_per_page
        commands_on_page = self.command_list[start_index:end_index]

        # Pair commands into columns
        column_pairs = list(zip_longest(commands_on_page[:len(commands_on_page)//2],
                                        commands_on_page[len(commands_on_page)//2:],
                                        fillvalue=("", "")))

        for pair in column_pairs:
            (cmd1, desc1), (cmd2, desc2) = pair
            table.add_row(cmd1, desc1, cmd2, desc2)

        return table

    def render_panel(self):
        """Render the help commands in a Rich Panel."""
        table = self.create_table()
        total_pages = (len(self.command_list) - 1) // self.commands_per_page + 1
        return Panel(
            table,
            title=f"[bold green]Help Menu[/bold green] - Page {self.current_page + 1}/{total_pages}",
            border_style="blue",
        )

    def run(self):
        """Run the interactive command navigator."""
        bindings = KeyBindings()

        # Quit on 'q'
        @bindings.add("q")
        def quit_handler(event):
            event.app.exit()

        # Scroll down on 'j'
        @bindings.add("j")
        def next_page(event):
            if self.current_page < (len(self.command_list) - 1) // self.commands_per_page:
                self.current_page += 1
                live.update(self.render_panel())

        # Scroll up on 'k'
        @bindings.add("k")
        def previous_page(event):
            if self.current_page > 0:
                self.current_page -= 1
                live.update(self.render_panel())

        # Create the application
        app = Application(
            key_bindings=bindings,
            full_screen=True,  # Enables full terminal screen rendering
        )

        with Live(self.render_panel(), refresh_per_second=4, screen=True) as live:
            app.run()  # Runs the prompt_toolkit application


# Example usage
if __name__ == "__main__":
    from itertools import zip_longest
    commands = {
        f"cmd{i}": f"Description for command {i}" for i in range(1, 51)  # Example: 50 commands
    }

    navigator = CommandNavigator(commands)
    navigator.run()

