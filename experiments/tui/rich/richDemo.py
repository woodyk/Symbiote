#!/usr/bin/env python3
#
# richDemo.py

from rich import print
from rich.align import Align
from rich.bar import Bar
from rich.color import Color
from rich.columns import Columns
from rich.console import Console, ConsoleOptions
from rich.emoji import Emoji
from rich.highlighter import Highlighter
from rich.json import JSON
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.markup import escape
from rich.measure import Measurement
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import Progress, BarColumn, TaskID
from rich.progress_bar import ProgressBar
from rich.prompt import Prompt, Confirm
from rich.protocol import is_renderable
from rich.rule import Rule
from rich.segment import Segment
from rich.spinner import Spinner
from rich.status import Status
from rich.style import Style
from rich.styled import Styled
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.traceback import Traceback
from rich.tree import Tree
from rich.abc import RichRenderable

# Create a Console for output
console = Console()


# 1. Align
console.print(Align("Aligned Center Text", align="center"))
console.print(Align("Aligned Right Text", align="right"))

# 2. Bar
bar = Bar(size=100, begin=25, end=75, color="blue")
console.print(bar)

# 3. Color
console.print(Color.parse("red"), Color.parse("#00ff00"))

# 4. Columns
columns = Columns(["Column 1", "Column 2", "Column 3"], align="center")
console.print(columns)

# 5. Emoji
try:
    console.print(Emoji("smile"))  # Should work
    console.print(Emoji(":rocket:"))  # Might not work, use fallback
except Exception:
    console.print("🚀 (Fallback Rocket Emoji)")  # Use Unicode directly

# 6. Highlighter
class CustomHighlighter(Highlighter):
    def highlight(self, text):
        text.stylize("bold red", 0, 5)

console.print(CustomHighlighter()("Highlight Me!"))

# 7. JSON
console.print(JSON('{"key": "value", "list": [1, 2, 3]}'))

# 8. Layout
layout = Layout()
layout.split_column(
    Layout(name="top"),
    Layout(name="bottom"),
)
layout["top"].split_row(Layout(name="left"), Layout(name="right"))
console.print(layout)

# 9. Live
with Live(console=console, refresh_per_second=4):
    for i in range(10):
        console.print(f"Live Update {i}")

# 10. Logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")
logger.debug("This is a debug message")

# 11. Markdown
markdown = Markdown("# Markdown Heading\n**Bold Text**\n*Italic Text*")
console.print(markdown)

# 12. Markup
console.print(f"[bold magenta]This is bold and magenta[/]")

# 14. Padding
console.print(Padding("This text has padding!", (2, 4)))

# 15. Panel
panel = Panel("Panel Content", title="Panel Title")
console.print(panel)

# 16. Pretty
data = {"key": "value", "list": [1, 2, 3]}
console.print(Pretty(data))

# 17. Progress Bar
progress_bar = ProgressBar()
console.print(progress_bar)

# 18. Progress
with Progress() as progress:
    task = progress.add_task("[cyan]Downloading...", total=100)
    for i in range(100):
        progress.update(task, advance=1)

# 19. Prompt
name = Prompt.ask("What is your name?")
console.print(f"Hello, {name}!")

# 20. Protocol
console.print(is_renderable("String"))

# 21. Rule
console.print(Rule("Section Break"))

# 22. Segment
segment = Segment("Segment Example")
console.print(segment)

# 23. Spinner
with console.status("[bold green]Working..."):
    console.print(Spinner("dots"))

# 24. Status
with Status("Processing...", console=console):
    console.print("Doing something...")

# 25. Style
style = Style(color="red", bold=True)
console.print("Styled Text", style=style)

# 26. Styled
styled = Styled("Styled Renderable", style="green")
console.print(styled)

# 27. Syntax
syntax = Syntax("def foo(): pass", "python", theme="monokai", line_numbers=True)
console.print(syntax)

# 28. Table
table = Table(title="Table Example")
table.add_column("Column 1", justify="center")
table.add_column("Column 2", justify="right")
table.add_row("Data 1", "Data 2")
console.print(table)

# 29. Text
text = Text("Rich Text Example")
text.stylize("bold", 0, 4)
console.print(text)

# 30. Theme
theme = Theme({"success": "green", "error": "red"})
console.print("Success message!")

# 31. Traceback
try:
    1 / 0
except ZeroDivisionError:
    console.print(Traceback())

# 32. Tree
tree = Tree("Root Node")
tree.add("Branch 1").add("Leaf 1")
tree.add("Branch 2").add("Leaf 2")
console.print(tree)

# 33. RichRenderable
class MyRenderable(RichRenderable):
    def __rich_console__(self, console, options):
        yield Text("Custom RichRenderable Implementation")

console.print(MyRenderable())

