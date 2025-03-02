#!/usr/bin/env python3
#
# pt_pager.py

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, ScrollablePane, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame
from pygments import highlight
from pygments.lexers import guess_lexer_for_filename, MarkdownLexer
from pygments.formatters import TerminalFormatter
import os

try:
    from markdown import markdown
except ImportError:
    markdown = None  # Markdown support requires the `markdown` package.


class Pager:
    """
    A full-featured pager for displaying long text in the console.
    Supports terminal colors, Markdown rendering, and syntax highlighting.
    """

    def __init__(self, text: str = None, file: str = None, title: str = "Pager", render_markdown: bool = False):
        """
        Initialize the pager with text content or a file.

        Args:
            text (str): The text to display in the pager.
            file (str): The file to load and display.
            title (str): Title for the pager display.
            render_markdown (bool): Whether to render Markdown content.
        """
        if text is None and file is None:
            raise ValueError("Either 'text' or 'file' must be provided.")

        self.file = file
        self.title = title
        self.render_markdown = render_markdown
        self.style = Style.from_dict({
            "frame.border": "#00ff00",
            "title": "bold underline",
            "content": "#ffffff bg:#000000",
        })

        # Load text
        if file:
            text = self._load_file(file)

        # Process text for Markdown or syntax highlighting
        if render_markdown and markdown:
            self.content = self._render_markdown(text)
        elif file:
            self.content = self._highlight_syntax(file, text)
        else:
            self.content = text

        # Create components
        self.buffer = Buffer(document=Document(self.content, cursor_position=0), read_only=True)
        self.text_window = Window(BufferControl(buffer=self.buffer), style="class:content", wrap_lines=True)
        self.key_bindings = self._create_key_bindings()
        self.application = self._create_application()

    def _load_file(self, file_path: str) -> str:
        """
        Load the content of a file.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: Content of the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _render_markdown(self, text: str) -> str:
        """
        Render Markdown text to plain terminal-formatted text.

        Args:
            text (str): Markdown text.

        Returns:
            str: Rendered plain text with ANSI colors.
        """
        from prompt_toolkit.formatted_text import HTML
        if not markdown:
            raise ImportError("The 'markdown' package is required for Markdown rendering.")
        html_content = markdown(text)
        return HTML(html_content)

    def _highlight_syntax(self, filename: str, text: str) -> str:
        """
        Apply syntax highlighting to the text using Pygments.

        Args:
            filename (str): Filename to guess the syntax.
            text (str): Content to highlight.

        Returns:
            str: Syntax-highlighted content.
        """
        try:
            lexer = guess_lexer_for_filename(filename, text)
            return highlight(text, lexer, TerminalFormatter())
        except Exception:
            # Fallback to plain text if syntax highlighting fails
            return text

    def _create_key_bindings(self) -> KeyBindings:
        """
        Create key bindings for scrolling and exiting the pager.

        Returns:
            KeyBindings: Configured key bindings.
        """
        kb = KeyBindings()

        @kb.add("q")
        def _(event):
            "Exit the pager."
            event.app.exit()

        @kb.add("up")
        @kb.add("k")
        def _(event):
            "Scroll up."
            buffer = event.app.current_buffer
            buffer.cursor_up(count=1)

        @kb.add("down")
        @kb.add("j")
        def _(event):
            "Scroll down."
            buffer = event.app.current_buffer
            buffer.cursor_down(count=1)

        @kb.add("pageup")
        def _(event):
            "Scroll up a page."
            buffer = event.app.current_buffer
            buffer.cursor_up(count=10)

        @kb.add("pagedown")
        def _(event):
            "Scroll down a page."
            buffer = event.app.current_buffer
            buffer.cursor_down(count=10)

        @kb.add("home")
        def _(event):
            "Go to the top."
            buffer = event.app.current_buffer
            buffer.cursor_position = 0

        @kb.add("end")
        def _(event):
            "Go to the bottom."
            buffer = event.app.current_buffer
            buffer.cursor_position = len(buffer.text)

        return kb

    def _create_application(self) -> Application:
        """
        Create the pager application.

        Returns:
            Application: A Prompt Toolkit application for the pager.
        """
        # Wrap the text window in a ScrollablePane and Frame
        frame = Frame(
            ScrollablePane(self.text_window),
            title=self.title,
            style="class:frame.border",
        )
        layout = Layout(HSplit([frame]))
        return Application(layout=layout, key_bindings=self.key_bindings, full_screen=True, style=self.style)

    def run(self):
        """
        Run the pager application.
        """
        self.application.run()


if __name__ == "__main__":
    # Example usage

    # 1. Plain text
    #plain_text = "This is a simple text example.\n" * 50
    #pager1 = Pager(text=plain_text, title="Plain Text Pager")
    #pager1.run()

    # 2. Markdown file
    markdown_example = """
    # Header 1
    ## Header 2
    - Item 1
    - Item 2
    ```
    Code block example
    ```
    """
    #pager2 = Pager(text=markdown_example, title="Markdown Example", render_markdown=True)
    #pager2.run()

    # 3. Syntax highlighting (Python file example)
    python_code = """
    def hello_world():
        print("Hello, World!")

    hello_world()
    """
    pager3 = Pager(title="Python Code Example", file="./curses_panel.py")
    pager3.run()

