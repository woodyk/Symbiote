#!/usr/bin/env python3
#
# tt.py
from prompt_toolkit.shortcuts import button_dialog
from prompt_toolkit.styles import Style

# Define custom styles
custom_style = Style.from_dict({
    "dialog": "bg:#0f111a",
    "button": "bg:#0f111a",
    "dialog.body": "bg:#0f111a",
    "dialog shadow": "bg:#0f111a",
    "dialog.frame.label": "bg:#0f111a",
})

# Create a simple dialog with options
dialog = button_dialog(
    title="Select an option",
    text="Please choose an option:",
    buttons=[
        ("Option 1", 1),
        ("Option 2", 2),
        ("Option 3", 3),
    ],
    style=custom_style
)

# Run the dialog and get the selected option
selected = dialog.run()
print("Selected:", selected)

