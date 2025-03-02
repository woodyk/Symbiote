#!/usr/bin/env python3
#
# richHighlight.py

from rich.highlighter import Highlighter, ISO8601Highlighter, JSONHighlighter, NullHighlighter, RegexHighlighter, ReprHighlighter
from rich.console import Console
from rich.text import Text

# Initialize Console
console = Console()

# Custom Highlighter: Example of extending the abstract Highlighter class
class CustomHighlighter(Highlighter):
    def highlight(self, text):
        """Highlight all occurrences of 'custom'."""
        text.highlight_regex(r"(custom)", "bold red")

# Initialize custom highlighter
custom_highlighter = CustomHighlighter()

# Example text
example_text = "This is a custom highlighter example. Let's highlight the word 'custom'."

# Apply custom highlighter
console.rule("CustomHighlighter Example")
console.print(custom_highlighter(example_text))

# ISO8601Highlighter
iso_highlighter = ISO8601Highlighter()
iso_text = Text("The event starts on 2024-11-15T10:00:00Z and ends on 2024-11-15T18:00:00Z.")
console.rule("ISO8601Highlighter Example")
console.print(iso_highlighter(iso_text))

# JSONHighlighter
json_highlighter = JSONHighlighter()
json_text = Text('{"name": "John", "age": 30, "city": "New York"}')
console.rule("JSONHighlighter Example")
console.print(json_highlighter(json_text))

# NullHighlighter
null_highlighter = NullHighlighter()
null_text = Text("This text will not be highlighted at all.")
console.rule("NullHighlighter Example")
console.print(null_highlighter(null_text))

# RegexHighlighter: Example of creating a regex-based highlighter
class EmailHighlighter(RegexHighlighter):
    base_style = "bold green"
    highlights = [r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"]

email_highlighter = EmailHighlighter()
email_text = Text("Contact us at support@example.com or admin@example.org for more info.")
console.rule("RegexHighlighter Example")
console.print(email_highlighter(email_text))

# ReprHighlighter
repr_highlighter = ReprHighlighter()
repr_text = Text("User(name='John', age=30, active=True)")
console.rule("ReprHighlighter Example")
console.print(repr_highlighter(repr_text))

# Summary of different highlighters
console.rule("Summary of Highlighters")
console.print(
    Text(
        "1. CustomHighlighter: Highlights the word 'custom' in bold red.\n"
        "2. ISO8601Highlighter: Highlights ISO8601 date-time strings.\n"
        "3. JSONHighlighter: Highlights JSON text.\n"
        "4. NullHighlighter: Does not apply any highlighting.\n"
        "5. RegexHighlighter: Highlights emails in bold green.\n"
        "6. ReprHighlighter: Highlights Python __repr__ outputs."
    )
)

