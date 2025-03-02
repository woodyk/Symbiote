#!/usr/bin/env python3
#
# richText.py

from rich.console import Console
from rich.text import Text

# Initialize Console
console = Console()

# Example 1: Basic Styling
console.rule("Example 1: Basic Styling")
text = Text("Hello, Rich!", style="bold red")
console.print(text)

# Example 2: Text Alignment
console.rule("Example 2: Text Alignment")
aligned_text = Text("Centered Text", style="cyan")
aligned_text.align("center", width=50)
console.print(aligned_text)

right_aligned = Text("Right Aligned", style="green")
right_aligned.align("right", width=50)
console.print(right_aligned)

# Example 3: Text Wrapping and Overflow
console.rule("Example 3: Text Wrapping and Overflow")
wrapped_text = Text(
    "This is a long line of text that will demonstrate wrapping and overflow.",
    style="magenta",
)
wrapped_text.wrap(console, width=40, overflow="ellipsis")
console.print(wrapped_text)

# Example 4: Dynamic Text Building
console.rule("Example 4: Dynamic Text Building")
dynamic_text = Text("Rich ", style="bold blue")
dynamic_text.append("is ", style="italic green")
dynamic_text.append("awesome!", style="bold magenta")
console.print(dynamic_text)

# Example 5: Highlighting Words
console.rule("Example 5: Highlighting Words")
highlighted_text = Text("Highlight specific words in this text.", style="dim")
highlighted_text.highlight_words(["Highlight", "words"], style="bold yellow")
console.print(highlighted_text)

# Example 6: Highlighting with Regex
console.rule("Example 6: Highlighting with Regex")
regex_text = Text("Email me at contact@example.com.", style="dim")
regex_text.highlight_regex(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", style="underline green")
console.print(regex_text)

# Example 7: Tabs and Indentation
console.rule("Example 7: Tabs and Indentation")
tabbed_text = Text("This\tis\ta\ttest.", style="blue")
tabbed_text.expand_tabs(tab_size=4)
console.print(tabbed_text)

indented_text = Text("Indented\n  Content\n    Deeper")
indented_text = indented_text.with_indent_guides(indent_size=2, character="│", style="dim green")
console.print(indented_text)

# Example 8: Truncation
console.rule("Example 8: Truncation")
long_text = Text("This text is very long and needs to be truncated.", style="red")
long_text.truncate(30, overflow="ellipsis")
console.print(long_text)

# Example 9: Combining and Joining
console.rule("Example 9: Combining and Joining")
part1 = Text("Rich ", style="bold cyan")
part2 = Text("Text ", style="bold magenta")
part3 = Text("Features", style="bold yellow")
combined_text = part1 + part2 + part3
console.print(combined_text)

separator = Text(", ", style="dim")
joined_text = separator.join([Text("One"), Text("Two"), Text("Three")])
console.print(joined_text)

# Example 10: Styled Assembly
console.rule("Example 10: Styled Assembly")
assembled_text = Text.assemble(
    ("Rich ", "bold blue"),
    ("Text ", "italic green"),
    ("Assembly!", "bold red underline"),
)
console.print(assembled_text)

# Example 11: Rendering ANSI Escape Codes
console.rule("Example 11: Rendering ANSI Escape Codes")
ansi_text = Text.from_ansi("\033[1;31mHello, ANSI Text!\033[0m")
console.print(ansi_text)

# Example 12: Rendering Markup
console.rule("Example 12: Rendering Markup")
markup_text = Text.from_markup("[bold magenta]Rich[/bold magenta] [green]Text[/green] [yellow]Markup[/yellow]")
console.print(markup_text)

