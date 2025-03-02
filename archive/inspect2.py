#!/usr/bin/env python3
#
# inspect2.py

from rich import inspect
from rich.console import Console

console = Console()

def recursive_inspect(obj, depth=1, max_depth=3, obj_name=None):
    """
    Recursively inspects an object up to a specified depth.

    Parameters:
    - obj: The object to inspect.
    - depth: Current depth of recursion.
    - max_depth: Maximum depth to inspect.
    """
    all_objs = list({**globals(), **locals()}.keys())

    if depth > max_depth:
        return

    # Display the object attributes using rich.inspect
    console.print(f"\n[bold]Inspecting level {depth}[/bold] - {obj.__class__.__name__}")
    inspect(obj, console=console, title=obj_name, all=False, methods=True, help=True)

    # Iterate through the attributes in dir(obj)
    for attr_name in dir(obj):
        if not attr_name.startswith("__"):  # Ignore dunder attributes
            attr = getattr(obj, attr_name, None)
            
            # Recur if the attribute is a class instance or a complex object
            if isinstance(attr, (object, list, dict, set, tuple)) and not isinstance(attr, (str, int, float, bool)):
                recursive_inspect(attr, depth + 1, max_depth, obj_name=attr_name)

recursive_inspect("rich.markdown", max_depth=50, obj_name="rich.markdown")

