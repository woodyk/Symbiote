#!/usr/bin/env python3
#
# richMarkup.py

from rich.console import Console
from rich.markup import escape, render

# Initialize the console
console = Console()

# 1. **Basic Markup**
console.rule("[bold cyan]Basic Markup Examples[/bold cyan]")
console.print("[bold red]This text is bold and red[/bold red]")
console.print("[italic green]Italic and green[/italic green]")
console.print("[underline blue]Underlined and blue[/underline blue]")

# 2. **Combining Styles**
console.rule("[bold cyan]Combining Styles[/bold cyan]")
console.print("[bold red on yellow]Bold red text on yellow background[/bold red on yellow]")
console.print("[blink magenta]Blinking magenta text[/blink magenta]")

# 3. **Shorthand Style Closing**
console.rule("[bold cyan]Shorthand Style Closing[/bold cyan]")
console.print("[bold red]Bold and red[/] Normal text")

# 4. **Overlapping Markup Tags**
console.rule("[bold cyan]Overlapping Markup Tags[/bold cyan]")
console.print("[bold]Bold [italic]Bold and Italic[/bold] Italic[/italic]")

# 5. **Links**
console.rule("[bold cyan]Links[/bold cyan]")
console.print("[link=https://www.python.org]Python Official Site[/link]")
console.print("[bold link=https://github.com]GitHub[/bold link=https://github.com]")

# 6. **Escaping Markup**
console.rule("[bold cyan]Escaping Markup[/bold cyan]")
raw_text = "This is [bold red]red text[/bold red] without rendering."
escaped_text = escape(raw_text)
console.print(f"Escaped: {escaped_text}")
console.print(f"Rendered: {raw_text}")

# 7. **Dynamic Markup**
console.rule("[bold cyan]Dynamic Markup[/bold cyan]")
user_input = "[blink]Danger![/blink]"
safe_input = escape(user_input)  # Prevents markup from being executed
console.print(f"User Input Rendered Safely: {safe_input}")

# 8. **Emojis**
console.rule("[bold cyan]Emojis[/bold cyan]")
console.print(":rocket: [bold]Rich is taking off![/bold] :sparkles:")
console.print(":red_heart-emoji: Emoji Variant: Full Color")
console.print(":red_heart-text: Emoji Variant: Text")

# 9. **Rendering Text from Markup**
console.rule("[bold cyan]Rendering Markup to Text Object[/bold cyan]")
text_obj = render("[bold blue]This is a rendered Text object[/bold blue]")
console.print(text_obj)

# 10. **Disabling Markup**
console.rule("[bold cyan]Disabling Markup[/bold cyan]")
console.print("[bold red]This won't render markup[/bold red]", markup=False)

# 11. **Using Markup Inside Panels and Tables**
console.rule("[bold cyan]Markup in Panels and Tables[/bold cyan]")
from rich.panel import Panel
from rich.table import Table

panel = Panel("[bold magenta]Rich Panel with Markup[/bold magenta]")
console.print(panel)

table = Table(title="[bold cyan]Table with Markup[/bold cyan]")
table.add_column("[bold green]Column 1[/bold green]")
table.add_column("[bold yellow]Column 2[/bold yellow]")
table.add_row("[italic red]Row 1, Col 1[/italic red]", "[underline]Row 1, Col 2[/underline]")
table.add_row("[dim blue]Row 2, Col 1[/dim blue]", "[reverse]Row 2, Col 2[/reverse]")
console.print(table)

# 12. **Complex Nested Markup**
console.rule("[bold cyan]Complex Nested Markup[/bold cyan]")
console.print("[bold red]Error: [italic]Invalid value for parameter[/italic][/bold red]")

# 13. **Mixing Markup with Dynamic Content**
console.rule("[bold cyan]Dynamic Content with Markup[/bold cyan]")
for i in range(5):
    console.print(f"[bold cyan]Item {i}[/bold cyan]: [italic green]{i * 2}[/italic green]")

console.rule("[bold green]Rich Markup Demonstration Complete[/bold green]")

