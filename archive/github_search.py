#!/usr/bin/env python3
#
# github_search.py

import requests
import json
import subprocess
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import Completer, Completion
from typing import Callable
import subprocess
import os
import subprocess
from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import VSplit, FloatContainer, HSplit, Window
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import SearchToolbar, TextArea, Frame
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.application.current import get_app
import types

console = Console()

class GitHubSearch:
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.text-match+json",
        }
        self.endpoints = {}
        self.fetch_endpoints()

    def fetch_endpoints(self):
        """Fetch all available endpoints from the GitHub API root."""
        response = requests.get(self.base_url, headers=self.headers)
        if response.status_code == 200:
            self.endpoints = response.json()
            console.log("Fetched available endpoints.", style="bold green")
        else:
            console.log("Failed to fetch endpoints.", style="bold red")
            exit(1)

    def search(self, endpoint_key, query, **params):
        """Perform a search on a specific endpoint."""
        if endpoint_key not in self.endpoints:
            console.log(f"Invalid endpoint key: {endpoint_key}", style="bold red")
            return {}

        url = self.endpoints[endpoint_key].replace("{query}", query)
        for placeholder in ["{&page,per_page,sort,order}", "{repository_id}"]:
            url = url.replace(placeholder, "")

        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            console.log("Rate limit exceeded. Try again later.", style="bold yellow")
        else:
            console.log(f"Failed request: {response.status_code}", style="bold red")
        return {}

    def search_all(self, query):
        """Search across all supported endpoints."""
        results = {}
        for key, url in self.endpoints.items():
            if "search" in key:  # Restrict to search-related endpoints
                console.log(f"Searching: {key}")
                results[key] = self.search(key, query)
        return results
    def display_results(self, results, format="table"):
        """Display results in different formats."""
        if not results:
            console.log("No results to display.", style="bold yellow")
            return

        if format == "json":
            console.print_json(json.dumps(results, indent=2))
        elif format == "markdown":
            console.print(Markdown(json.dumps(results, indent=2)))
        elif format == "table":
            for key, data in results.items():
                if data and "items" in data:
                    table = Table(title=f"Results for {key}", expand=True)
                    table.add_column("Name")
                    table.add_column("Description")
                    table.add_column("URL")
                    for item in data["items"]:
                        table.add_row(
                            item.get("name", "N/A"),
                            item.get("description", "N/A"),
                            item.get("html_url", "N/A"),
                        )
                    console.print(table)
                else:
                    console.log(f"No results for {key}.", style="bold yellow")
        else:
            console.log(f"Invalid format: {format}", style="bold red")

    def save_results(self, results, filename="results.json"):
        """Save results to a file."""
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        console.log(f"Results saved to {filename}", style="bold green")

class GitHubCLIApp:
    def __init__(self):
        self.searcher = GitHubSearch()
        self.commands = {}
        self.register_builtin_commands()

        # Initialize UI components
        self.output_field = TextArea(
            "Welcome to the GitHub CLI App!\n", style="class:output-field", scrollbar=True
        )
        self.input_field = TextArea(
            height=10,
            prompt=">>> ",
            style="class:input-field",
            multiline=False,
            wrap_lines=False,
        )

        self.style = Style.from_dict({
            "frame.border": "fg:white",
            "frame.title": "fg:cyan",
            "content": "fg:white",
            "output-field": "#ffffff",
            "input-field": "bg:#000000 #ffffff",
            "line": "#004400"
        })

    def register_builtin_commands(self):
        """Register built-in commands."""
        self.register_command("/search", self.cmd_search, "/search <endpoint> <query> - Search a specific endpoint", 2)
        self.register_command("/list", self.cmd_list, "/list - List available endpoints", 0)
        self.register_command("/exit", self.cmd_exit, "/exit - Exit the application", 0)
        self.register_command("/add_command", self.cmd_add_command, "/add_command <name> <code> - Add a new command dynamically", 2)

    def register_command(self, name, handler, help_text, args):
        """Register a new command dynamically."""
        self.commands[name] = {
            "handler": handler,
            "help": help_text,
            "args": args,
        }

    def execute_command(self, user_input):
        """Execute a command."""
        parts = user_input.strip().split()
        if not parts:
            return "Invalid command."

        command = parts[0]
        args = parts[1:]

        if command not in self.commands:
            return f"Unknown command: {command}. Type /list for available commands."

        if len(args) < self.commands[command]["args"]:
            return f"Usage: {self.commands[command]['help']}"

        try:
            return self.commands[command]["handler"](*args)
        except Exception as e:
            return f"Error executing command: {e}"

    # Command Handlers
    def cmd_search(self, endpoint, query):
        """Handler for /search command."""
        results = self.searcher.search(endpoint, query)
        return str(results) if results else "No results found."

    def cmd_list(self):
        """Handler for /list command."""
        return "\n".join([f"{cmd}: {info['help']}" for cmd, info in self.commands.items()])

    def cmd_exit(self):
        """Handler for /exit command."""
        get_app().exit()
        return "Exiting application..."

    def cmd_add_command(self, name, code):
        """Handler for dynamically adding commands."""
        try:
            exec(f"def {name}(self, *args): {code}", globals())
            setattr(self.__class__, name, globals()[name])
            self.register_command(f"/{name}", getattr(self, name), f"/{name} - Dynamically added command", 0)
            return f"Command /{name} added successfully."
        except Exception as e:
            return f"Failed to add command: {e}"

    # Full-Screen Application
    def run(self):
        terminal_width, terminal_height = os.get_terminal_size()
        width_percent = 0.4
        height_percent = 0.4
        calculated_width = int(terminal_width * width_percent)
        calculated_height = int(terminal_height * height_percent)

        # Add a vertical split
        hsplit_content = HSplit([
            self.output_field,
            Window(height=1, char="-", style="class:line"),
            self.input_field,
        ])

        framed_hsplit = Frame(
            hsplit_content,
            title="GitHub CLI",
            style="class:frame",
            height=calculated_height
        )

        def accept(buff):
            user_input = self.input_field.text
            self.input_field.text = ""  # Clear input field
            output = self.execute_command(user_input)

            # Append output to the output field
            new_text = self.output_field.text + f">>> {user_input}\n{output}\n"
            self.output_field.buffer.document = Document(
                text=new_text, cursor_position=len(new_text)
            )

        # Key bindings
        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):
            event.app.exit()

        self.input_field.accept_handler = accept

        layout = Layout(framed_hsplit, focused_element=self.input_field)

        app = Application(
            layout=layout,
            key_bindings=kb,
            style=self.style,
            mouse_support=True,
            full_screen=True,
        )
        app.run()

class GitHubCLI:
    def __init__(self):
        self.searcher = GitHubSearch()
        self.session = PromptSession()
        self.style = Style.from_dict({
            "prompt": "#87CEEB bold",
        })
        self.commands = {}
        self.register_builtin_commands()

    def register_builtin_commands(self):
        """Register built-in commands."""
        self.register_command(
            "/search",
            self.cmd_search,
            "/search <endpoint> <query> - Search a specific endpoint",
            args=2,
        )
        self.register_command(
            "/search_all",
            self.cmd_search_all,
            "/search_all <query> - Search all available endpoints",
            args=1,
        )
        self.register_command(
            "/save",
            self.cmd_save,
            "/save <query> <filename> - Save results to a file",
            args=2,
        )
        self.register_command(
            "/list",
            self.list_endpoints,
            "/list - List all available endpoints",
            args=0,
        )
        self.register_command(
            "/help",
            self.display_help,
            "/help [command] - Display help for all commands or a specific command",
            args=0,
        )
        self.register_command(
            "/exit",
            self.cmd_exit,
            "/exit - Exit the application",
            args=0,
        )
        self.register_command(
            "/shell",
            self.exec_command,
            "/shell <bash_command> - Run shell command",
            args=3,
        )

    def register_command(self, name: str, handler: Callable, help_text: str, args: int):
        """Register a new command with the CLI."""
        console.log(name, handler, help_text, args)
        console.log(args)
        self.commands[name] = {
            "handler": handler,
            "help": help_text,
            "args": args,
        }

    def display_help(self, command=None):
        """Display help for all commands or a specific command."""
        if command and command in self.commands:
            self.searcher.console.print(self.commands[command]["help"], style="bold yellow")
        else:
            self.searcher.console.print("Available commands:", style="bold yellow")
            for cmd, details in self.commands.items():
                self.searcher.console.print(f"{details['help']}", style="yellow")

    def validate_command(self, command, args):
        """Validate that the correct number of arguments are provided for a command."""
        if command not in self.commands:
            self.searcher.console.log(f"Invalid command: {command}", style="bold red")
            return False
        expected_args = self.commands[command]["args"]
        if len(args) < expected_args:
            self.searcher.console.log(
                f"Insufficient arguments for {command}. Expected {expected_args}, got {len(args)}.",
                style="bold red",
            )
            self.display_help(command)
            return False
        return True

    def main_menu(self):
        """Main interactive menu."""
        while True:
            try:
                command_input = self.session.prompt(
                    "[GitHubSearch]> ",
                    style=self.style,
                    completer=CommandCompleter(self.commands, self.searcher.endpoints),
                )
                if not command_input.strip():
                    continue

                # Parse command and arguments
                parts = command_input.split()
                command = parts[0]
                args = []
                args.append("\n".join(args))
                #args = parts[1:]

                if not self.validate_command(command, args):
                    continue

                # Execute the command
                handler = self.commands[command]["handler"]
                handler(*args)
            except KeyboardInterrupt:
                console.log("\nExiting application.", style="bold cyan")
                break
            except Exception as e:
                console.log(f"An error occurred: {str(e)}", style="bold red")

    # Command Handlers
    def cmd_search(self, endpoint, query):
        results = self.searcher.search(endpoint, query)
        self.searcher.display_results(results)

    def cmd_search_all(self, query):
        results = self.searcher.search_all(query)
        self.searcher.display_results(results)

    def cmd_save(self, query, filename):
        results = self.searcher.search_all(query)
        self.searcher.save_results(results, filename)

    def list_endpoints(self):
        endpoints = self.searcher.endpoints
        if not endpoints:
            self.searcher.console.log("No endpoints available. Fetch endpoints first.", style="bold red")
            return

        table = Table(title="Available Endpoints", expand=True)
        table.add_column("Endpoint Key", style="cyan", no_wrap=True)
        table.add_column("URL", style="green")

        for key, url in endpoints.items():
            table.add_row(key, url)

        self.searcher.console.print(table)

    def cmd_exit(self):
        self.searcher.console.log("Exiting application.", style="bold cyan")
        exit(0)

    def exec_command(self, shell_command):
        command_output = subprocess.check_output('/bin/bash','-c', shell_command).decode('utf-8').strip()
        console.print(Syntax(command_output, "bash"))

class CommandCompleter(Completer):
    global console
    """Custom completer for command and argument suggestions."""

    def __init__(self, commands, endpoints):
        self.commands = commands
        self.endpoints = endpoints

    def get_completions(self, document, complete_event):
        text = document.text
        if not text.startswith("/"):
            return

        parts = text.split()
        if len(parts) == 1:
            # Suggest commands
            for cmd in self.commands.keys():
                if cmd.startswith(parts[0]):
                    yield Completion(cmd, start_position=-len(parts[0]))
        elif len(parts) == 2:
            # Suggest arguments (e.g., endpoint keys)
            command = parts[0]
            if command == "/search" and parts[1]:
                for key in self.endpoints.keys():
                    if key.startswith(parts[1]):
                        yield Completion(key, start_position=-len(parts[1]))


if __name__ == "__main__":
    #cli = GitHubCLI()
    #cli.main_menu()
    app = GitHubCLIApp()
    app.run()

