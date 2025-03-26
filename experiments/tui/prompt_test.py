#!/usr/bin/env python3
#
# prompt_test.py

from prompt_toolkit import Application
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.styles import Style
from prompt_toolkit.layout.margins import ScrollbarMargin

from rich.console import Console
from rich.text import Text

import sys

# Define a custom exception for exiting the application
class ExitCommand(Exception):
    pass



def main():
    # Initialize console for rich rendering
    console = Console()

    # List to store messages
    messages = []

    # Settings dictionary with default values
    settings = {
        "theme": "dark",
        "font_size": "12",
        "language": "en",
    }

    # Function to get the formatted text for the message area
    def get_message_text():
        # Combine all messages into a single ANSI formatted text
        return ANSI(''.join(messages))

    # Create the message display area with a scrollbar
    message_area = Window(
        content=FormattedTextControl(get_message_text, focusable=False),
        wrap_lines=True,
        always_hide_cursor=True,
        right_margins=[ScrollbarMargin(display_arrows=True)]
    )

    # Create the input area
    input_area = TextArea(
        height=1,
        prompt='> ',
        multiline=False,
        wrap_lines=False
    )

    # Create the layout as a vertical split: messages above, input below
    body = HSplit([
        message_area,
        input_area,
    ])

    # Define custom styles
    style = Style.from_dict({
        'scrollbar.background': '#555555',
        'scrollbar.button': '#888888',
    })

    # Create the key bindings
    kb = KeyBindings()

    # Command registry
    # Each command maps to a dict with 'func' and 'help'
    commands = {}

    def register_command(name, help_text):
        """
        Decorator to register a new command with a help description.
        Usage:
            @register_command('/command_name', 'Description of the command')
            def command(args):
                # Your code here
                return 'Output message'
        """
        def decorator(func):
            commands[name] = {'func': func, 'help': help_text}
            return func
        return decorator

    # Example command: /hello
    @register_command('/hello', 'Says hello.')
    def hello_command(args):
        rich_text = Text("Hello!", style="bold magenta")
        # Capture the rich output as ANSI text
        with console.capture() as capture:
            console.print(rich_text)
        return capture.get()

    # Example command: /echo
    @register_command('/echo', 'Echoes the provided arguments. Usage: /echo <message>')
    def echo_command(args):
        if not args:
            return 'Usage: /echo <message>\n'
        return ' '.join(args) + '\n'

    # Example command: /add
    @register_command('/add', 'Adds the provided numbers. Usage: /add 1 2 3.5')
    def add_command(args):
        if not args:
            return 'Usage: /add num1 num2 ...\n'
        try:
            numbers = list(map(float, args))
            result = sum(numbers)
            return f'The sum is: {result}\n'
        except ValueError:
            return 'Please provide valid numbers to add.\n'

    # Command to exit the application
    @register_command('/exit', 'Exits the application.')
    @register_command('/quit', 'Exits the application.')
    def exit_command(args):
        raise ExitCommand()

    # Command to display help
    @register_command('/help', 'Displays available commands. Usage: /help [command]')
    def help_command(args):
        if args:
            cmd = args[0]
            if cmd in commands:
                return f"{cmd}: {commands[cmd]['help']}\n"
            else:
                return f"No help available for unknown command: {cmd}\n"
        else:
            help_text = "Available commands:\n"
            for cmd, info in commands.items():
                help_text += f"  {cmd}: {info['help']}\n"
            return help_text

    # Command to view and set settings
    @register_command('/settings', 'View or modify settings. Usage: /settings, /settings get <key>, /settings set <key> <value>')
    def settings_command(args):
        if not args:
            # Display all settings
            if not settings:
                return 'No settings available.\n'
            help_text = "Current settings:\n"
            for key, value in settings.items():
                help_text += f"  {key}: {value}\n"
            return help_text
        elif len(args) >= 2:
            subcommand = args[0].lower()
            if subcommand == 'get':
                key = args[1]
                if key in settings:
                    return f"{key}: {settings[key]}\n"
                else:
                    return f"Setting '{key}' does not exist.\n"
            elif subcommand == 'set':
                if len(args) < 3:
                    return "Usage: /settings set <key> <value>\n"
                key = args[1]
                value = ' '.join(args[2:])
                settings[key] = value
                return f"Setting '{key}' updated to '{value}'.\n"
            else:
                return "Invalid subcommand. Usage: /settings, /settings get <key>, /settings set <key> <value>\n"
        else:
            return "Invalid usage. Usage: /settings, /settings get <key>, /settings set <key> <value>\n"

    # Handle Enter key
    @kb.add('enter')
    def _(event):
        user_input = input_area.text.strip()
        if not user_input:
            return  # Ignore empty input
        if user_input.startswith('/'):
            # Command mode
            parts = user_input.split()
            cmd_name = parts[0]
            args = parts[1:]
            if cmd_name in commands:
                try:
                    result = commands[cmd_name]['func'](args)
                    messages.append(result)
                except ExitCommand:
                    # Exit the application
                    event.app.exit()
                except Exception as e:
                    messages.append(f'Error executing {cmd_name}: {str(e)}\n')
            else:
                messages.append(f'Unknown command: {cmd_name}\n')
        else:
            # Regular message
            messages.append(f'You said: {user_input}\n')
        input_area.text = ''
        # Refresh the UI
        event.app.invalidate()

    # Handle Ctrl+C to exit cleanly
    @kb.add('c-c')
    def _(event):
        event.app.exit()

    # Create the application
    application = Application(
        layout=Layout(body, focused_element=input_area),
        key_bindings=kb,
        style=style,
        full_screen=True
    )

    try:
        # Run the application
        application.run()
    except (KeyboardInterrupt, ExitCommand):
        # Handle any remaining exit scenarios
        sys.exit(0)

if __name__ == '__main__':
    main()

