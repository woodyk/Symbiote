#!/usr/bin/env python3
#
# richPretty.py

from rich.console import Console
from rich.pretty import Pretty, pprint, pretty_repr, traverse, Node
from rich.panel import Panel
from rich import pretty

# Initialize Console
console = Console()

# Example Data for Demonstration
nested_data = {
    "name": "Rich Pretty",
    "version": "1.0",
    "features": ["color", "pretty-print", "interactive"],
    "nested": {
        "a": [1, 2, 3],
        "b": {"x": "hello", "y": "world"},
        "c": (4, 5, 6),
    },
    "long_string": "This is a very long string that will demonstrate truncation.",
}

# Example 1: Basic Pretty Rendering
console.rule("Basic Pretty Example")
pretty_obj = Pretty(nested_data, indent_size=2, max_length=10)
console.print(pretty_obj)

# Example 2: Pretty Printing with pprint
console.rule("Using pprint Function")
pprint(nested_data, max_length=20, max_string=30, max_depth=2)

# Example 3: Custom Pretty Repr
console.rule("Using pretty_repr Function")
pretty_repr_string = pretty_repr(
    nested_data,
    max_width=60,
    indent_size=4,
    max_length=10,
    max_string=50,
    expand_all=False,
)
console.print(Panel(pretty_repr_string, title="Pretty Repr Output"))

# Example 4: Traversing an Object
console.rule("Traversing Nested Object")
traversed_tree = traverse(nested_data, max_depth=2)
console.print("Traversal Output:")
for token in traversed_tree.iter_tokens():
    console.print(token)

# Example 5: Using Node Class
console.rule("Custom Node Example")
node = Node(
    key_repr="root",
    value_repr="Root Node",
    children=[
        Node(key_repr="child1", value_repr="Value1", last=False),
        Node(key_repr="child2", value_repr="Value2", last=True),
    ],
    open_brace="{",
    close_brace="}",
)
console.print(Panel(node.render(), title="Custom Node"))

# Example 6: Installing Pretty Printing
console.rule("Installing Pretty Printing in REPL")
pretty.install(console=console, max_depth=3, indent_guides=True, max_string=30)

# Example 7: Large Data with Limits
console.rule("Large Data Example with Limits")
large_data = {"key": [f"value {i}" for i in range(50)]}
pprint(large_data, max_length=10, max_depth=1)

# Example 8: Automatic Highlighting
console.rule("Pretty Print with Highlighting")
pprint(
    nested_data,
    console=console,
    max_length=10,
    max_string=30,
    expand_all=False,
    indent_guides=True,
)

# Example 9: Inline Render with Pretty
console.rule("Inline Pretty Render")
inline_pretty = Pretty(
    {"inline": [1, 2, 3, {"nested": "data"}]}, indent_size=2, max_depth=3
)
console.print(inline_pretty)

# Example 10: Handling Long Strings
console.rule("Handling Long Strings")
long_string_data = {"long_text": "This is an example of a very long string." * 10}
pprint(long_string_data, max_string=50)

# Summary of Features
console.rule("Summary of rich.pretty Features")
summary = """
1. Pretty Rendering: Use Pretty to create renderable representations.
2. pprint: Pretty print directly to the console.
3. pretty_repr: Generate custom repr strings.
4. traverse: Traverse objects to generate tree-like structures.
5. Node: Create custom repr trees for objects.
6. Install: Install automatic pretty printing in REPL.
7. Large Data Handling: Manage limits on depth, length, and string size.
8. Highlighting: Use automatic highlighting with pprint.
9. Inline Rendering: Inline Pretty instances in Rich layouts.
10. Long Strings: Manage truncation of long strings.
"""
console.print(Panel(summary, title="Features Summary"))

