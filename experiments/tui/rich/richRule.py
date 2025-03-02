#!/usr/bin/env python3
#
# richRule.py

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

# Initialize Console
console = Console()

# Example 1: Basic Rule
console.print("Example 1: Basic Rule")
basic_rule = Rule()
console.print(basic_rule)

# Example 2: Rule with a Title
console.print("Example 2: Rule with Title")
title_rule = Rule(title="Section Title")
console.print(title_rule)

# Example 3: Rule with Title Aligned Left
console.print("Example 3: Rule with Left-Aligned Title")
left_aligned_rule = Rule(title="Left Aligned Title", align="left")
console.print(left_aligned_rule)

# Example 4: Rule with Title Aligned Right
console.print("Example 4: Rule with Right-Aligned Title")
right_aligned_rule = Rule(title="Right Aligned Title", align="right")
console.print(right_aligned_rule)

# Example 5: Rule with Custom Characters
console.print("Example 5: Rule with Custom Characters")
custom_characters_rule = Rule(title="Custom Characters", characters="*~*")
console.print(custom_characters_rule)

# Example 6: Rule with Custom Style
console.print("Example 6: Rule with Custom Style")
styled_rule = Rule(title="Styled Rule", style="bold green")
console.print(styled_rule)

# Example 7: Rule with Title as a `Text` Instance
console.print("Example 7: Rule with Title as a Text Instance")
text_title = Text("Rich Rule Example", style="bold underline magenta")
text_rule = Rule(title=text_title)
console.print(text_rule)

# Example 8: Rule with No Title
console.print("Example 8: Rule with No Title")
no_title_rule = Rule()
console.print(no_title_rule)

# Example 9: Rule Without Line Break (Custom End Character)
console.print("Example 9: Rule Without Line Break")
no_line_break_rule = Rule(title="No Line Break", end="")
console.print(no_line_break_rule)

# Example 10: Combining Rules with Other Renderables
console.print("Example 10: Combining Rules with Other Renderables")
console.print(Rule(title="Start of Section", style="bold red"))
console.print("[bold]Some content here...[/bold]")
console.print(Rule(title="End of Section", style="bold red"))

# Example 11: Rule with Dynamic Title
console.print("Example 11: Rule with Dynamic Title")
dynamic_title = "Dynamic Title: Example 11"
dynamic_rule = Rule(title=dynamic_title)
console.print(dynamic_rule)

# Example 12: Rule with Multiline Text
console.print("Example 12: Rule with Multiline Text")
multiline_title = Text("Multiline\nTitle", justify="center", style="cyan")
multiline_rule = Rule(title=multiline_title)
console.print(multiline_rule)

# Summary of Features
console.print("Summary of rich.rule Features")
summary_rule = Rule(title="Summary of Features", style="bold yellow")
console.print(summary_rule)

summary = """
1. Basic Rule: A plain horizontal line with no title.
2. Title: Add a title to the rule for context or labels.
3. Title Alignment: Align the title left, center, or right.
4. Custom Characters: Use unique characters for the line.
5. Custom Styles: Apply styles to the rule and its title.
6. `Text` Title: Use `Text` for styled or complex titles.
7. No Title: Use a rule without a title.
8. No Line Break: Prevent automatic line breaks after the rule.
9. Combination: Combine rules with other content.
10. Dynamic Title: Dynamically generate rule titles.
11. Multiline Title: Use multiline or styled titles for complex displays.
"""
console.print(summary)

