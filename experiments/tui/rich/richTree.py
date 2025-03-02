#!/usr/bin/env python3
#
# richTree.py

from rich.tree import Tree
from rich.console import Console
from rich.text import Text

# Initialize Console
console = Console()

# Example 1: Basic Tree
console.rule("Example 1: Basic Tree")
tree1 = Tree("Root Node")
tree1.add("Child 1")
tree1.add("Child 2")
console.print(tree1)

# Example 2: Nested Trees
console.rule("Example 2: Nested Trees")
tree2 = Tree("Root Node")
branch = tree2.add("Parent")
branch.add("Child 1").add("Grandchild 1").add("Great-grandchild 1")
branch.add("Child 2")
branch2 = tree2.add("Another Parent")
branch2.add("Another Child")
console.print(tree2)

# Example 3: Tree with Custom Styles
console.rule("Example 3: Tree with Custom Styles")
tree3 = Tree(
    "Root Node (Bold and Cyan)",
    style="bold cyan",
    guide_style="bold magenta",
)
branch = tree3.add("Parent Node (Green)", style="green")
branch.add("Child Node 1 (Yellow)", style="yellow")
branch.add("Child Node 2 (Red)", style="red")
console.print(tree3)

# Example 4: Highlighting Tree Nodes
console.rule("Example 4: Highlighting Tree Nodes")
tree4 = Tree("Root Node", highlight=True)
tree4.add("[bold yellow]Highlighted Child 1[/bold yellow]")
tree4.add("[italic green]Highlighted Child 2[/italic green]")
console.print(tree4)

# Example 5: Collapsed Tree
console.rule("Example 5: Collapsed Tree")
tree5 = Tree("Root Node", expanded=False)
tree5.add("Child 1")
tree5.add("Child 2")
console.print(tree5)

# Example 6: Tree with Hidden Root
console.rule("Example 6: Tree with Hidden Root")
tree6 = Tree("Hidden Root Node", hide_root=True)
branch = tree6.add("Visible Parent Node")
branch.add("Visible Child Node")
console.print(tree6)

# Example 7: Trees with Rich Renderables
console.rule("Example 7: Trees with Rich Renderables")
from rich.table import Table

# Create a table renderable
table = Table(title="Sample Table")
table.add_column("Name", justify="left")
table.add_column("Value", justify="right")
table.add_row("Item 1", "Value 1")
table.add_row("Item 2", "Value 2")

# Add table as a tree node
tree7 = Tree("Root Node")
tree7.add(table)
console.print(tree7)

# Example 8: Mixed Content Tree
console.rule("Example 8: Mixed Content Tree")
tree8 = Tree(Text("Root Node", style="bold underline magenta"))
tree8.add(Text("Plain Text Child", style="dim"))
branch = tree8.add(Text("Styled Branch", style="green bold"))
branch.add(Text("Nested Child", style="italic red"))
console.print(tree8)

# Example 9: Directory-like Tree
console.rule("Example 9: Directory-like Tree")
dir_tree = Tree("Root Directory")
dir_tree.add("file1.txt")
subdir = dir_tree.add("subdir")
subdir.add("file2.txt")
subdir.add("file3.txt")
console.print(dir_tree)

# Example 10: Tree with Complex Styling
console.rule("Example 10: Tree with Complex Styling")
tree10 = Tree(
    "Stylized Root Node",
    style="bold on blue",
    guide_style="dim underline",
)
branch1 = tree10.add("Branch 1", style="on green")
branch1.add("Leaf 1", style="yellow")
branch1.add("Leaf 2", style="cyan")
branch2 = tree10.add("Branch 2", style="on magenta")
branch2.add("Leaf 3", style="white on red")
console.print(tree10)

