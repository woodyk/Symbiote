#!/usr/bin/env python3
#
# richPrompt.py

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm

# Initialize Console
console = Console()

# Example 1: Basic String Prompt
console.rule("Basic String Prompt")
name = Prompt.ask("Enter your name")
console.print(f"Hello, [bold green]{name}[/bold green]!")

# Example 2: Prompt with Default Value
console.rule("Prompt with Default Value")
favorite_color = Prompt.ask("What's your favorite color?", default="Blue")
console.print(f"Your favorite color is [bold blue]{favorite_color}[/bold blue].")

# Example 3: Prompt with Choices
console.rule("Prompt with Choices")
planet = Prompt.ask(
    "Choose your home planet", choices=["Earth", "Mars", "Venus"], default="Earth"
)
console.print(f"You chose [bold green]{planet}[/bold green].")

# Example 4: Integer Prompt
console.rule("Integer Prompt")
age = IntPrompt.ask("How old are you?")
console.print(f"You are [bold cyan]{age}[/bold cyan] years old.")

# Example 5: Float Prompt
console.rule("Float Prompt")
temperature = FloatPrompt.ask("Enter the temperature")
console.print(f"The temperature is [bold magenta]{temperature}°C[/bold magenta].")

# Example 6: Confirmation Prompt
console.rule("Confirmation Prompt")
is_sunny = Confirm.ask("Is it sunny outside?")
console.print(f"Sunny status: [bold yellow]{'Yes' if is_sunny else 'No'}[/bold yellow]")

# Example 7: Prompt with Custom Validation
console.rule("Custom Validation Prompt")


def validate_email(value: str) -> bool:
    if "@" in value and "." in value:
        return True
    raise ValueError("Invalid email address.")


email = Prompt.ask("Enter your email", default="example@example.com", choices=None)
while True:
    try:
        if validate_email(email):
            break
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        email = Prompt.ask("Enter your email again")
console.print(f"Your email is [bold green]{email}[/bold green].")

# Example 8: Password Prompt
console.rule("Password Prompt")
password = Prompt.ask("Enter your password", password=True)
console.print("[bold red]Password securely entered![/bold red]")

# Example 9: Handling Invalid Responses
console.rule("Handling Invalid Responses")


class CustomPrompt(Prompt):
    def process_response(self, value: str):
        if value.lower() not in ["apple", "banana", "cherry"]:
            raise ValueError("Please choose a valid fruit: apple, banana, cherry.")
        return value.capitalize()

try:
    fruit = CustomPrompt.ask("Choose a fruit")
    console.print(f"You chose: [bold green]{fruit}[/bold green]")
except Exception as e:
    print(f"Error: {e)"

# Example 10: Using Prompt in a Script
console.rule("Prompt in a Script")
if Confirm.ask("Do you want to continue with this script?"):
    console.print("[bold green]Continuing...[/bold green]")
else:
    console.print("[bold red]Exiting...[/bold red]")

# Example 11: Customizing Prompt Display
console.rule("Customizing Prompt Display")
customized_prompt = Prompt.ask(
    "Choose an option",
    choices=["Option 1", "Option 2", "Option 3"],
    show_choices=False,
    show_default=False,
)
console.print(f"You selected: [bold green]{customized_prompt}[/bold green]")

# Example 12: Error Handling with InvalidResponse
console.rule("Error Handling with InvalidResponse")


def validate_positive_number(value: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError("Number must be positive.")
    return number


while True:
    try:
        positive_number = IntPrompt.ask("Enter a positive number")
        validate_positive_number(positive_number)
        break
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")

console.print(f"Positive number: [bold cyan]{positive_number}[/bold cyan]")

# Example 13: Prompt with Stream
console.rule("Prompt with Stream")
import io

stream_input = io.StringIO("Mars\n")
with console.capture() as capture:
    planet_with_stream = Prompt.ask("Enter your favorite planet", stream=stream_input)
console.print(f"Captured input: [bold green]{planet_with_stream}[/bold green]")

# Summary of Features
console.rule("Summary of rich.prompt Features")
summary = """
1. Prompt.ask: Basic string input with optional defaults.
2. IntPrompt: Prompt for integer input.
3. FloatPrompt: Prompt for float input.
4. Confirm: Yes/No confirmation prompts.
5. Validation: Custom validation for user input.
6. Password: Hidden input for secure prompts.
7. Choices: Loop until a valid choice is selected.
8. Custom Prompts: Inherit and extend the Prompt class.
9. Error Handling: Handle invalid responses gracefully.
10. Streams: Use streams for scripted input.
"""
console.print(summary)

