#!/usr/bin/env python3
#
# tt.py

from rich.markdown import Markdown
from rich import print as rp
from rich import inspect
import sys

file = sys.argv[1]

# example markdown text
markdown_text = """
# Hello World

This is an **example** of using Markdown in the terminal with `rich`.
"""

rp(markdown_text, ":vampire:", locals())
# render markdown text in the terminal
md = Markdown(markdown_text)
print(md)

my_list = ["foo", "bar"]
inspect(my_list, methods=True)

from rich.console import Console
from rich.markdown import Markdown

console = Console()
with open(file) as readme:
    markdown = Markdown(readme.read())
console.print(markdown)

from rich.console import Console
from rich.syntax import Syntax

my_code = '''
def iter_first_last(values: Iterable[T]) -> Iterable[Tuple[bool, bool, T]]:
    """Iterate and generate a tuple with a flag for first and last value."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    first = True
    for value in iter_values:
        yield first, False, previous_value
        first = False
        previous_value = value
    yield first, True, previous_value
'''
syntax = Syntax(my_code, "python", theme="monokai", line_numbers=True)
console = Console()
console.print(syntax)

