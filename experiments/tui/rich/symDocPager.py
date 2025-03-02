#!/usr/bin/env python3
#
# symDocPager.py

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
import time


class DocumentPager:
    def __init__(self, content: str, content_type: str = "plain", width: int = None):
        """
        Initializes the Document Pager.

        Args:
            content (str): The document content.
            content_type (str): The type of content ('plain', 'markdown', 'syntax').
            width (int): Optional width for rendering. Defaults to terminal width.
        """
        self.console = Console()
        self.content = content
        self.content_type = content_type
        self.width = width or self.console.size.width

    def render_content(self) -> list:
        """
        Renders the content based on the type.

        Returns:
            list: The rendered content as a list of lines.
        """
        with self.console.capture() as capture:
            if self.content_type == "markdown":
                self.console.print(Markdown(self.content), width=self.width)
            elif self.content_type == "syntax":
                syntax = Syntax(self.content, "python", theme="monokai", line_numbers=True)
                self.console.print(syntax, width=self.width)
            else:  # Plain text
                self.console.print(self.content, width=self.width)

        # Capture output and split into lines
        return capture.get().splitlines()

    def slide_in(self, lines: list, direction: str = "top", delay: float = 0.05):
        """
        Slide in the content with an animation.

        Args:
            lines (list): List of strings to display.
            direction (str): Animation direction ('top' or 'left').
            delay (float): Delay between animation frames.
        """
        if direction == "left":
            for step in range(self.width, -1, -1):
                self.console.clear()
                for line in lines:
                    self.console.print(" " * step + line)
                time.sleep(delay)
        elif direction == "top":
            for step in range(len(lines) + 1):
                self.console.clear()
                for line in lines[:step]:
                    self.console.print(line.center(self.width))
                time.sleep(delay)

    def display_paged(self, lines_per_page: int = 20, direction: str = "top", delay: float = 0.05):
        """
        Displays the content page-by-page with sliding animation.

        Args:
            lines_per_page (int): Number of lines to display per page.
            direction (str): Animation direction ('top' or 'left').
            delay (float): Delay between animation frames.
        """
        lines = self.render_content()
        total_pages = (len(lines) + lines_per_page - 1) // lines_per_page

        for page in range(total_pages):
            start = page * lines_per_page
            end = start + lines_per_page
            page_lines = lines[start:end]

            # Slide in the current page
            self.slide_in(page_lines, direction=direction, delay=delay)

            # Display navigation
            self.console.print(f"[bold blue]Page {page + 1}/{total_pages}[/bold blue]")
            if page < total_pages - 1:
                self.console.print("[bold yellow]Press ENTER or type command to continue...[/bold yellow]")
                input()
            else:
                self.console.print("[bold green]End of document.[/bold green]")

            self.console.clear()


# Example Usage
if __name__ == "__main__":
    # Example content (Markdown)
    markdown_content = """
# Document Pager Example
This is an example of a **Document Pager** with sliding animations.
You can render Markdown, Syntax Highlighting, or Plain Text!

## Features
- Slide-in animations from top or left
- Supports multiple content types
- Displays content page-by-page

Enjoy the terminal magic with **Rich**!
    """

    # Example content (Syntax Highlighting)
    code_content = """
def example_function(arg1, arg2):
    \"\"\"This is a simple function example.\"\"\"
    return arg1 + arg2
    """

    # Initialize pager
    pager = DocumentPager(content=markdown_content, content_type="markdown")
    pager.display_paged(lines_per_page=10, direction="top", delay=0.05)

    pager = DocumentPager(content=code_content, content_type="syntax")
    pager.display_paged(lines_per_page=10, direction="left", delay=0.05)

