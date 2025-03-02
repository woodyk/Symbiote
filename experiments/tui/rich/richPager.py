#!/usr/bin/env python3
#
# richPager.py

import sys
import re
from typing import List, Optional
import readchar
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.align import Align
from rich.prompt import Prompt
from rich.theme import Theme
from rich.traceback import install

# Optional: Install rich traceback for better exception handling
#install()

class Pager:
    """
    A full-screen pager that allows scrolling, searching, and rendering content
    in plain text or markdown.
    """
    def __init__(self, content: str, render_mode: str = 'plain'):
        """
        Initialize the Pager.

        Args:
            content (str): The content to display in the pager.
            render_mode (str): 'plain' for plain text or 'markdown' for markdown rendering.
        """
        self.console = Console()
        self.content = content.splitlines()
        self.render_mode = render_mode.lower()
        self.top_line = 0  # Index of the top visible line
        self.search_term: Optional[str] = None
        self.matches: List[int] = []  # Line indices with matches
        self.highlight_style = "bold yellow on blue"
        self.height = self.console.size.height
        self.width = self.console.size.width
        self.quit = False
        self.in_search = False
        self.search_query = ""
        self.max_top_line = max(len(self.content) - self.height, 0)

    def render_content(self):
        """
        Render the current view of the content, with highlights if searching.

        Returns:
            Panel: The rendered panel containing the visible content.
        """
        self.height = self.console.size.height
        self.width = self.console.size.width

        # Calculate the ending line index based on the top line and the height
        bottom_line = self.top_line + self.height
        bottom_line = min(bottom_line, len(self.content))  # Clamp to max content length
        self.max_top_line = max(len(self.content) - self.height, 0)

        # Track the current visible lines
        visible_content = self.content[self.top_line:bottom_line]
        
        header_text = Text(f"t:{self.top_line}, b:{bottom_line}, h:{self.height}, dl:{len(self.content)}, mtl:{self.max_top_line}, vl:{len(visible_content)}")


        # Handle rendering mode
        if self.render_mode == 'markdown':
            body_text = Markdown("\n".join(visible_content))
        else:
            body_text = Text("\n".join(visible_content))

        # Highlight search matches if applicable
        if self.search_term:
            pattern = re.compile(re.escape(self.search_term), re.IGNORECASE)
            highlighted_lines = []
            for idx, line in enumerate(visible_content):
                absolute_idx = self.top_line + idx  # Map visible index to absolute index
                if absolute_idx in self.matches:
                    matches = list(pattern.finditer(line))
                    if matches:
                        if self.render_mode == 'markdown':
                            # Highlight matches in markdown content
                            highlighted_line = line
                            for match in reversed(matches):  # Process from the end to avoid index shifting
                                start, end = match.span()
                                highlighted_line = (
                                    highlighted_line[:start]
                                    + f"[{self.highlight_style}]"
                                    + highlighted_line[start:end]
                                    + "[/]"
                                    + highlighted_line[end:]
                                )
                            highlighted_lines.append(highlighted_line)
                        else:
                            # Highlight matches in plain text content
                            text = Text(line)
                            for match in matches:
                                text.stylize(self.highlight_style, match.start(), match.end())
                            highlighted_lines.append(text)
                    else:
                        highlighted_lines.append(line)
                else:
                    highlighted_lines.append(line)

            # Update renderable based on highlighting
            if self.render_mode == 'markdown':
                renderable = Markdown("\n".join(str(line) for line in highlighted_lines))
            else:
                renderable = Text("\n".join(str(line) for line in highlighted_lines))

            # Wrap the content in a panel for display
            panel = Panel(renderable, style="none")

        group = []
        group.append(header_text)
        group.append(body_text)

        panel_group = Group(*group)
        panel = Panel(panel_group, height=self.height)
        return panel

    def search(self, term: str):
        """
        Search for the term in the content and store matching line indices.

        Args:
            term (str): The search term to look for.
        """
        self.search_term = term
        self.matches = [i for i, line in enumerate(self.content) if re.search(re.escape(term), line, re.IGNORECASE)]
        if not self.matches:
            self.console.print(f"No matches found for '{term}'", style="bold red")
        else:
            # Reset top_line to first match if not already visible
            first_match = self.matches[0]
            if not (self.top_line <= first_match < self.top_line + self.height):
                self.top_line = min(first_match, self.max_top_line)

    def clear_search(self):
        """Clear the current search term and matches."""
        self.search_term = None
        self.matches = []

    def handle_keypress(self, key: str):
        """
        Handle a single keypress event to navigate or interact with the pager.

        Args:
            key (str): The key that was pressed.
        """
        if self.in_search:
            # Search mode input handling...
            pass
        else:
            # Normal mode navigation
            if key == 'q':  # Quit the pager
                self.quit = True
            elif key in ['j', readchar.key.DOWN]:  # Scroll down by one line
                if self.top_line <= len(self.content):  
                    self.top_line += 1
            elif key in ['k', readchar.key.UP]:  # Scroll up by one line
                if self.top_line > 0:
                    self.top_line -= 1
            elif key == readchar.key.PAGE_DOWN:  # Page down
                if self.top_line + self.height <= self.max_top_line:
                    self.top_line = min(self.top_line + self.height, self.max_top_line)
            elif key == readchar.key.PAGE_UP:  # Page up
                if self.top_line > 0:
                    self.top_line = max(self.top_line - self.height, 0)
            elif key == '/':  # Enter search mode
                self.in_search = True
                self.search_query = ""
                self.console.print("\rSearch: ", end='', style="bold yellow on black")
            elif key == 'n':  # Jump to next search match
                if self.matches:
                    current = next((i for i in self.matches if i > self.top_line), None)
                    if current is not None:
                        self.top_line = min(current, self.max_top_line)
            elif key == 'N':  # Jump to previous search match
                if self.matches:
                    current = next((i for i in reversed(self.matches) if i < self.top_line), None)
                    if current is not None:
                        self.top_line = max(current - self.height + 1, 0)

    def run(self):
        """
        Run the pager, handling rendering and keypress events.
        """
        with Live(self.render_content(), refresh_per_second=10, console=self.console, screen=True) as live:
            while not self.quit:
                try:
                    key = readchar.readkey()
                    self.handle_keypress(key)
                    if key:
                        pass
                        #live.update(Text(f"{self.top_line}, {self.height}"))
                except KeyboardInterrupt:
                    self.quit = True
                    break

                try:
                    live.update(self.render_content())
                except Exception as e:
                    console.log(f"Exception: {e}")


def main():
    """
    Demonstrate the Pager with sample content.
    """
    sample_text = """
# Rich Console Pager Example

This is an example of a full-screen pager using [bold green]Rich[/bold green]. You can scroll through the content using the arrow keys or `Page Up` and `Page Down`. Press `/` to search for a term, `n` to jump to the next match, `N` to jump to the previous match, and `q` to quit the pager.

## Features

- **Scrolling**: Use `↑`, `↓`, `Page Up`, `Page Down` to navigate.
- **Search**: Press `/` and enter a search term to highlight matches.
- **Rendering Modes**: Toggle between plain text and markdown.
- **Highlighting**: Search results are highlighted for easy identification.
- **Exit**: Press `q` to exit the pager.

## Sample Content

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus imperdiet, nulla et dictum interdum, nisi lorem egestas odio, vitae scelerisque enim ligula venenatis dolor.
Maecenas nisl est, ultrices nec congue eget, auctor vitae massa. Fusce luctus vestibulum augue ut aliquet. Nunc sagittis dictum nisi, sed ullamcorper ipsum dignissim ac. In at libero sed nunc venenatis imperdiet sed ornare turpis.

## Search Example

Try searching for the word "sed" to see how search functionality works. All instances of "sed" will be highlighted.

## Markdown Rendering

The pager supports [bold]markdown[/bold] rendering, allowing for rich text formatting within the content.

### Subsection

- Bullet Point 1
- Bullet Point 2
- Bullet Point 3

## End of Example

Thank you for using the Rich Console Pager!
"""

    pager = Pager(content=sample_text, render_mode='markdown')
    pager.run()
    console = Console()
    console.print("[bold green]Exited the pager. Goodbye![/bold green]")


if __name__ == "__main__":
    main()

