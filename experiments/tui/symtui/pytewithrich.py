#!/usr/bin/env python3
#
# pytewithrich.py

import os
import pty
import select
import signal
import sys
from rich.console import Console
from rich.text import Text
import pyte


class TerminalWrapper:
    def __init__(self, cols=80, rows=24):
        self.cols = cols
        self.rows = rows
        self.console = Console()
        self.screen = pyte.HistoryScreen(cols, rows, history=1000)  # Add history for scrolling.
        self.stream = pyte.ByteStream(self.screen)
        self.master_fd = None
        self.child_pid = None

    def start_shell(self, shell="/bin/bash"):
        """Start a pseudo-terminal running the specified shell."""
        self.child_pid, self.master_fd = pty.fork()
        if self.child_pid == 0:  # Child process
            os.environ["TERM"] = "xterm-256color"
            os.environ["COLUMNS"] = str(self.cols)
            os.environ["LINES"] = str(self.rows)
            os.execlp(shell, shell)
        # Parent process continues execution here

    def render_screen(self):
        """Render the terminal screen using rich, ensuring the prompt is at the bottom."""
        self.console.clear()  # Clear the console.
        # Display only the last `self.rows` lines of the screen buffer.
        lines = list(self.screen.display[-self.rows:])
        for line in lines:
            styled_line = Text.from_ansi(line)  # Convert ANSI to Rich Text.
            self.console.print(styled_line, end="\n")

    def run(self):
        """Run the terminal wrapper, handling I/O between user and shell."""
        try:
            while True:
                # Monitor master_fd and stdin for input
                rlist, _, _ = select.select([self.master_fd, sys.stdin], [], [])
                for fd in rlist:
                    if fd == self.master_fd:
                        # Read output from the shell
                        data = os.read(self.master_fd, 1024)
                        if not data:
                            return
                        self.stream.feed(data)
                        self.render_screen()
                    elif fd == sys.stdin:
                        # Forward user input to the shell
                        user_input = sys.stdin.read(1)
                        os.write(self.master_fd, user_input.encode())
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        """Terminate the child process and close the master_fd."""
        if self.child_pid:
            os.kill(self.child_pid, signal.SIGTERM)
        if self.master_fd:
            os.close(self.master_fd)
        self.console.print("\n[bold red]Terminal session ended.[/bold red]")


if __name__ == "__main__":
    wrapper = TerminalWrapper()
    wrapper.start_shell()
    wrapper.run()

