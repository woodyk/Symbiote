#!/usr/bin/env python3
#
# rich_pager.py

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from rich.protocol import is_renderable
from rich.measure import Measurement
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
import sys
import magic
import time
import os.path

class SymRichPager:
    def __init__(self, content):
        self.console = Console()
        self.live = None 
        render_object = self._parse_content(content)
        self.lines = render_object.splitlines()
        self.scroll_offset = 0
        self.page_height = self.console.size.height - 6
        self.header = Text(
            f"SymPager - [↑↓] Scroll, [Space] Page Down, [/] Search, [q] Quit"
        )

        self.footer = Text(
            f"[Controls: ↑↓ Scroll | Space: Page Down | /: Search | q: Quit]"
        )
        self.search_term = None
        self.search_bar = Panel(Text(f"[ {self.search_term} ]"))

    def _validate_file(self, content):
        """
        Check if the given path points to an existing file,
        handling ~ and relative paths appropriately.
        """
        if not isinstance(content, str) or not content.strip():
            return False  # Ensure the input is a valid non-empty string

        # Expand ~ to the user's home directory
        file_path = os.path.expanduser(content)

        # Convert relative path to absolute
        file_path = os.path.abspath(file_path)
        if os.path.isfile(file_path):
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type.startswith("text/"):
                with open(file_path, "r") as file:
                    file_contents = file.read()
                    return file_contents

        return False

    def _parse_content(self, content):
        """Parse content and render it appropriately."""
        file_contents = self._validate_file(content)
        if file_contents:
            return file_contents

        if hasattr(content, "__dict__"):  # If content is a Python object
            table = Table(title="Object Attributes")
            table.add_column("Attribute")
            table.add_column("Value")
            for key, value in vars(content).items():
                table.add_row(str(key), str(value))
            return str(table)

        elif isinstance(content, (list, set, tuple, dict)):  # Collections
            table = Table(title="Collection")
            table.add_column("Index/Key")
            table.add_column("Value")
            for key, value in (content.items() if isinstance(content, dict) else enumerate(content)):
                table.add_row(str(key), str(value))
            return str(table)

        elif isinstance(content, str):
            return content
            #return str(Syntax(content, "python", theme="monokai", line_numbers=True))
        else:
            raise ValueError("Unsupported content type")


    def _render_layout(self):
        """Render the current layout with header, content, and footer."""
        # Get visible lines
        visible_lines = self.lines[self.scroll_offset:self.scroll_offset + self.page_height]
        visible_content = "\n".join(visible_lines)
        content_panel = Panel(Syntax(visible_content, "python"))
        footer_panel = toggle_bar = Panel(self.footer)

        if self.search_term:
            toogle_bar = self.search_bar

        # Use Layout to structure header, content, and footer
        layout = Layout()
        layout.split_column(
            Layout(Panel(self.header), size=3),
            Layout(content_panel, size=self.page_height),
            Layout(toggle_bar, size=3),
        )
        return layout

    def display(self, content=None):
        """Main display loop with Live for dynamic rendering."""
        # Prep the pager_keys key bord mappings
        pager_keys = KeyBindings()

        
        #@pager_keys.add(Keys.Any)
        def clear_prompt(event):
            event.app.current_buffer.insert_text("e")
            event.app.current_buffer.validate_and_handle()

        # Quit
        @pager_keys.add("q", Keys.Escape)
        def _exit(event):
            self.running = False
            #sys.exit(0) 
            event.app.exit()
            #clear_prompt(event)

        # PageUP    
        @pager_keys.add(Keys.PageUp, Keys.ScrollUp)
        def _pgup(event):
            self.scroll_offset = max(
                self.scroll_offset - self.page_height,0
                )
            self.live.update(self._render_layout())
            #clear_prompt(event)

        # PageDown
        @pager_keys.add("space", Keys.ScrollDown, Keys.PageDown)
        def _pgdn(event):
            self.scroll_offset = min(
                self.scroll_offset + self.page_height,
                len(self.lines) - self.page_height
                )
            self.live.update(self._render_layout())
            #clear_prompt(event)

        # Right Arrow
        @pager_keys.add(Keys.Right)
        def _rght(event):
            self.live.update(self._render_layout())
            #clear_prompt(event)

        # Left Arrow
        @pager_keys.add(Keys.Left)
        def _left(event):
            self.live.update(self._render_layout())
            #clear_prompt(event)

        # Down Arrow
        @pager_keys.add(Keys.Down)
        def _down(event):
            if self.scroll_offset < len(self.lines) - self.page_height:
                self.scroll_offset += 1
            self.live.update(self._render_layout())
            #clear_prompt(event)

        # Up Arrow
        @pager_keys.add(Keys.Up)
        def _up(event):
            if self.scroll_offset > 0:
                self.scroll_offset -= 1
            self.live.update(self._render_layout())
            #clear_prompt(event)

        # Search
        @pager_keys.add("s")
        def _search(event):
            #search_term = self.console.input("Search: ")
            self.search_term = "hello"
            for i, line in enumerate(self.lines):
                if self.search_term in line:
                    self.scroll_offset = max(0, i - self.page_height // 2)
            self.live.update(self._render_layout())
            #clear_prompt(event)


        # Prepare our pager_session 
        pager_session = PromptSession(key_bindings=pager_keys)
        """
        self.live = Live(self._render_layout(),
                         console=self.console,
                         refresh_per_second=20,
                         screen=True)
        """

        with Live(self._render_layout(), console=self.console, refresh_per_second=20, screen=True) as self.live:
            while True:
                self.live.update(self._render_layout())
                u = pager_session.prompt(key_bindings=pager_keys)
                break


def main():
    if len(sys.argv) > 1:
        content = sys.argv[1]  
        pager = SymRichPager(content)
        pager.display()


if __name__ == "__main__":
    main()
