#!/usr/bin/env python3
#
# WindowManager.py

import os
import pty
import subprocess
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

# Initialize Rich console for rendering
console = Console()

# Window manager class
class WindowManager:
    def __init__(self):
        self.layout = Layout()
        self.panels = {}
        self.active_panel = "oooooo"

        # Initial layout structure
        self.layout.split(
            Layout(name="main"),  # Main area for output panels
            Layout(name="prompt", size=3),  # Bottom prompt bar
        )
        self.add_panel(self.active_panel) # ensure that at least one panel is up at init.
        self.update_prompt("Type your command below:")

    def add_panel(self, name: str):
        """Add a new panel."""
        if name in self.panels:
            return
        self.panels[name] = {"content": [], "panel": Panel("", title=name)}
        self.set_active_panel(name)
        self.arrange_panels()

    def update_panel(self, name: str, content: str):
        """Update the content of a panel."""
        print(name, content)
        if name in self.panels:
            self.panels[name]["content"].append(content)
            panel_content = "\n".join(self.panels[name]["content"])  # Keep last 20 lines
            self.panels[name]["panel"] = Panel(panel_content, title=name)
            self.arrange_panels()

    def arrange_panels(self):
        """Arrange panels dynamically."""
        panel_count = len(self.panels)
        if panel_count == 0:
            self.layout["main"].update(Panel("No active panels", border_style="red"))
        elif panel_count == 1:
            self.layout["main"].update(next(iter(self.panels.values()))["panel"])
        else:
            self.layout["main"].split_row(
                *[p["panel"] for p in self.panels.values()]
            )

    def set_active_panel(self, name: str):
        """Set the active panel."""
        if name in self.panels:
            self.active_panel = name

    def update_prompt(self, content: str):
        """Update the prompt panel."""
        self.layout["prompt"].update(Panel(content, border_style="green"))

# Custom shell application
class CustomShell:
    def __init__(self):
        self.window_manager = WindowManager()
        self.session = PromptSession(style=Style.from_dict({"prompt": "ansigreen bold"}))
        self.master_fd, self.slave_fd = pty.openpty()  # Create a pseudo-terminal
        self.bash_process = subprocess.Popen(
            ["/bin/bash"],
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            env={"PS1": ""},  # Disable Bash prompt
            text=True,
        )
        self.running = True
        self.current_input = ""

    def execute_command(self, command: str):
        """Execute a command using subprocess and display output."""
        if not self.window_manager.active_panel:
            self.window_manager.add_panel("Default")
            self.window_manager.set_active_panel("Default")

        try:
            # Run the command and capture output
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
            else:
                output = strderr.strip()

            self.window_manager.update_panel(self.window_manager.active_panel, result)
        except Exception as e:
            self.window_manager.update_panel(self.window_manager.active_panel, f"Error: {str(e)}")

    def process_input(self, user_input: str):
        """Process user input."""
        if user_input.startswith(":"):
            # Handle custom commands
            command, *args = user_input[1:].split()
            if command == "add":
                self.window_manager.add_panel(args[0])
            elif command == "remove":
                self.window_manager.remove_panel(args[0])
            elif command == "switch":
                self.window_manager.set_active_panel(args[0])
            elif command == "exit":
                self.running = False
            else:
                self.window_manager.update_prompt(f"Unknown command: {user_input}")
        else:
            # Execute as Bash command
            self.execute_command(user_input)

    def input_handler(self):
        """Create key bindings for real-time input."""
        kb = KeyBindings()

        @kb.add("<any>")
        def key_press(event):
            """Handle real-time keystrokes."""
            key = event.key_sequence[0].key
            if key == "c-d":  # Exit on Ctrl-D
                self.running = False
            elif key == "backspace":
                self.current_input = self.current_input[:-1]
            elif key == "enter":
                get_app().exit(result=self.current_input)
            elif len(key) == 1:
                self.current_input += key

            # Update the prompt panel with the current input
            self.window_manager.update_prompt("> " + self.current_input)

        return kb

    def run(self):
        """Run the custom shell."""
        key_bindings = self.input_handler()

        with Live(self.window_manager.layout, console=console, refresh_per_second=10):
            while self.running:
                try:
                    # Use PromptSession to capture user input
                    user_input = self.session.prompt("", key_bindings=key_bindings, default="")
                    self.process_input(user_input)
                    self.current_input = ""  # Clear input after execution
                    self.window_manager.update_prompt("Type your command below:")
                except (EOFError, KeyboardInterrupt):
                    self.running = False

# Entry point
if __name__ == "__main__":
    shell = CustomShell()
    shell.run()

