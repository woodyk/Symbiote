#!/usr/bin/env python3
#
# chatbot.py

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import checkboxlist_dialog, yes_no_dialog
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.widgets import Label, Box, TextArea, Frame, Button

# Define a style for the app
style = Style.from_dict({
    "prompt": "#00ff00 bold",
    "output": "#ffcc00 italic",
    "error": "red bold",
})

# Define available commands for the chatbot
commands = {
    "greet": "Say hello",
    "help": "Show this help menu",
    "calc": "Perform a calculation",
    "settings": "Adjust settings",
    "exit": "Exit the chatbot",
}

# Create a word completer for commands
command_completer = WordCompleter(list(commands.keys()), ignore_case=True)

# Define the main chatbot logic
def chatbot():
    session = PromptSession()
    print_formatted_text("Welcome to the Chatbot Showcase!", style=style)
    
    while True:
        # Prompt user for input
        user_input = session.prompt("> ", completer=command_completer, style=style)
        
        if user_input in commands:
            if user_input == "greet":
                print_formatted_text("Hello, User! How can I assist you?", style=style)
            elif user_input == "help":
                print_help_menu()
            elif user_input == "calc":
                perform_calculation(session)
            elif user_input == "settings":
                adjust_settings()
            elif user_input == "exit":
                print_formatted_text("Goodbye! Have a great day!", style=style)
                break
        else:
            print_formatted_text("Unknown command. Type 'help' to see available commands.", style=style)

# Display a help menu
def print_help_menu():
    print_formatted_text("Available Commands:", style=style)
    for cmd, desc in commands.items():
        print_formatted_text(f"{cmd}: {desc}", style=style)

# Perform a simple calculation
def perform_calculation(session):
    try:
        expr = session.prompt("Enter a mathematical expression: ", style=style)
        result = eval(expr)
        print_formatted_text(f"Result: {result}", style=style)
    except Exception as e:
        print_formatted_text(f"Error: {str(e)}", style=style)

# Adjust settings using dialogs
def adjust_settings():
    selected_options = checkboxlist_dialog(
        title="Settings",
        text="Choose your preferred settings:",
        values=[
            ("theme", "Dark Theme"),
            ("notifications", "Enable Notifications"),
            ("autosave", "Enable Autosave"),
        ],
    ).run()
    
    if selected_options:
        print_formatted_text(f"Settings adjusted: {', '.join(selected_options)}", style=style)
    else:
        print_formatted_text("No changes made to settings.", style=style)

# Entry point for the app
if __name__ == "__main__":
    chatbot()

