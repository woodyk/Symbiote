#!/usr/bin/env python3
#
# rich.py
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

def make_layout() -> Layout:
    """Create a layout with 3 columns and 3 rows where the middle column is wider."""
    layout = Layout()

    # Divide the layout into three columns
    layout.split_column(
        Layout(name="top", size=3),
        Layout(name="middle"),
        Layout(name="bottom", size=3)
    )

    # Adjust the width proportions
    # The ratios are adjusted to give the side columns less width
    for name in ["top", "bottom", "middle"]:
        layout[name].split_row(
            Layout(Panel(" ", title="Panel 1"), name=f"{name}_left", ratio=1),
            Layout(Panel(" ", title="Chat Display" if name == "middle" else "Panel 2"), name=f"{name}_center", ratio=3),
            Layout(Panel(" ", title="Panel 3"), name=f"{name}_right", ratio=1),
        )

    return layout

def chat_interface():
    console = Console()
    layout = make_layout()
    message_history = []  # Store chat history

    # Display layout
    console.print(layout)

    # Simulate chat interaction
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = f"Bot: Echoing - {user_input}"  # Bot's response
        message_history.append(f"You: {user_input}")  # Add user message to history
        message_history.append(response)  # Add bot response to history
        
        # Update chat display panel
        chat_content = "\n".join(message_history)  # Create chat content from history
        layout["middle_center"].update(Panel(chat_content, title="Chat Display"))
        console.print(layout)

if __name__ == "__main__":
    chat_interface()

