#!/usr/bin/env python3
#
# myapp.py

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import WordCompleter

# Command suggestions
command_completer = WordCompleter(["hello", "help", "exit", "rich", "prompt"], ignore_case=True)


class ChatApp:
    def __init__(self):
        self.console = Console()
        self.layout = self.create_layout()
        self.chat_history = []
        self.session = PromptSession("> ", completer=command_completer)

    def create_layout(self):
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["header"].update(Panel("ChatBot TUI", title="Header"))
        layout["body"].update(Panel("Welcome! Start typing below..."))
        layout["footer"].update(Panel("> "))
        return layout

    def run(self):
        with Live(self.layout, refresh_per_second=4, screen=True):
            while True:
                try:
                    user_input = self.session.prompt()
                    if user_input.lower() in ["exit", "quit"]:
                        break
                    self.handle_input(user_input)
                except KeyboardInterrupt:
                    break

    def handle_input(self, user_input):
        user_message = f"[bold cyan]You:[/bold cyan] {user_input}"
        bot_message = f"[bold green]Bot:[/bold green] I heard: {user_input}"
        self.chat_history.append(user_message)
        self.chat_history.append(bot_message)
        self.update_body()

    def update_body(self):
        history_text = "\n".join(self.chat_history[-10:])
        self.layout["body"].update(Panel(history_text, title="Conversation"))


if __name__ == "__main__":
    app = ChatApp()
    app.run()

