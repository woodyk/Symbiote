#!/usr/bin/env python3
#
# tt.py

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

console = Console()
import os
import sys

from rich import print
from rich.columns import Columns

directory = os.listdir("/Users/kato")
columns = Columns(directory, equal=True, expand=True)
print(columns)

# Create individual inner panels with titles
inner_panel_1 = Panel(
    Text("This is the content of the first inner panel."),
    title="Inner Panel 1",
    subtitle="Subtitle 1",
    border_style="blue",
    padding=(1, 2),
)

inner_panel_2 = Panel(
    Text("This is the content of the second inner panel."),
    title="Inner Panel 2",
    subtitle="Subtitle 2",
    border_style="green",
    padding=(1, 2),
)

# Arrange inner panels horizontally using Columns
inner_panels = Columns([inner_panel_1, inner_panel_2])

# Create an outer panel with the combined inner panels, and simulate no border
outer_panel = Panel(
    inner_panels,
    title="Outer Panel",
    subtitle="Main Container",
    border_style="",         # Use an empty string to simulate no border
    padding=(1, 2),
    style="on #f5f5f5"       # Light grey background color for the outer panel
)

# Print the outer panel containing the nested inner panels
console.print(outer_panel)
console.print()

t1 = Text("hello ther how are you")
t2 = Text("tell me more")

group = []
group.append(t1)
group.append(t2)

cont = Columns(group)

console.print(Panel(cont, expand=True))
