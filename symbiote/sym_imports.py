#!/usr/bin/env python3
#
# imports.py

sp = print
from rich.console import Console
from rich.pretty import pprint as pretty
from symbiote.sym_highlighter import SymHighlighter
sym_theme = SymHighlighter()
console = Console(highlighter=sym_theme, theme=sym_theme.theme)
#console = Console()
print = console.print
log = console.log

from rich import box
from rich import inspect
from rich.align import Align
from rich.box import SQUARE, DOUBLE
from rich.columns import Columns
from rich.console import Console
from rich.console import Group
from rich.highlighter import Highlighter, RegexHighlighter
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree

