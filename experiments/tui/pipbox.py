#!/usr/bin/env python3

import os
import multiprocessing
import subprocess
import signal
import time
from time import sleep, strftime
from rich.layout import Layout
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.align import Align
from rich.style import Style
from rich.text import Text
from prompt_toolkit import PromptSession
console = Console()

# Global variables for inter-process communication
box_process = None
box_pipe = None

def render_in_panel(data, location=2, border_color="cyan"):
    """
    Renders data in a rich Panel on the console based on the specified location.

    Args:
        data (str): The data to display in the panel.
        location (int): The location of the panel:
            1 = top left, 2 = top right, 3 = bottom left,
            4 = bottom right, 5 = center.
        border_color (str): The color of the panel border.
    """
    width, height = console.size.width, console.size.height

    # Default panel dimensions
    panel_width = width // 4
    panel_height = height // 4

    # Padding calculations based on location
    if location == 1:  # Top left
        pad_top, pad_left = 0, 0
    elif location == 2:  # Top right
        pad_top, pad_left = 0, width - panel_width
    elif location == 3:  # Bottom left
        pad_top, pad_left = height - panel_height, 0
    elif location == 4:  # Bottom right
        pad_top, pad_left = height - panel_height, width - panel_width
    elif location == 5:  # Center
        panel_width = width // 2
        panel_height = height // 2
        pad_top, pad_left = (height - panel_height) // 2, (width - panel_width) // 2
    else:  # Default to top right
        pad_top, pad_left = 0, width - panel_width

    # Padding for each direction
    padding = (pad_top, width - (pad_left + panel_width), height - (pad_top + panel_height), pad_left)

    # Render the Panel
    panel = Panel(
        data,
        padding=(0, 0),
        width=panel_width,
        height=panel_height,
    )
    console.print(panel)

def box_display(pipe, width, height):
    """
    The process to handle the live-refreshing box display.
    Args:
        pipe: The multiprocessing pipe for communication.
        width: Width of the box.
        height: Height of the box.
    """
    try:
        with Live(Align.center("", vertical="middle"), refresh_per_second=4, screen=True) as live:
            while True:
                if pipe.poll():
                    message = pipe.recv()
                    if message == "STOP":
                        break
                # Update content dynamically
                content = Text(f"Live Data:\n{strftime('%H:%M:%S')}", justify="center")
                live.update(Panel(content, width=width, height=height, title="Info Box"))
                time.sleep(0.1)  # Simulate refresh delay
    except KeyboardInterrupt:
        pass

def render_shell(command: str, width=None, height=None, padding_top=1, padding_right=1):
    """
    Renders a rich window box in the upper-right corner of the terminal displaying the output of a shell command.

    :param command: The shell command to execute.
    :param width: Width of the box (defaults to a quarter of the terminal width).
    :param height: Height of the box (defaults to a quarter of the terminal height).
    :param padding_top: Padding from the top of the terminal (default is 1 line).
    :param padding_right: Padding from the right of the terminal (default is 1 column).
    """
    terminal_width, terminal_height = console.size

    # Calculate panel dimensions
    box_width = width or terminal_width // 4
    box_height = height or terminal_height // 4

    # Execute the shell command and capture the output
    try:
        result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        output = f"Error executing command: {e}"

    # Create the panel
    panel = Panel(
        output,
        title=f"Command: {command}",
        title_align="left",
        border_style="bold cyan",
        width=box_width,
        height=box_height,
    )

    # Move cursor to the upper-right corner with padding and render the panel
    with console.capture() as capture:
        console.print(panel)
    content = capture.get()
    
    # Clear the terminal and position the panel
    print("\n" * padding_top, end="")
    print(" " * (terminal_width - box_width - padding_right), end="")
    print(content)
# Example usage:
# render_shell_command_box("ls -la", width=40, height=10)


def create_clock_widget():
    current_time = strftime("%H:%M:%S")
    align
    clock = Panel(
        Align.center(Text(current_time, style="bold green")),
        title="Clock",
        border_style="green",
        #box=DOUBLE
    )

    console.print(clock)


def start_box(width, height):
    """Start the box process."""
    global box_process, box_pipe
    if box_process and box_process.is_alive():
        print("[Box] Already running.")
        return
    parent_pipe, child_pipe = multiprocessing.Pipe()
    box_pipe = parent_pipe
    box_process = multiprocessing.Process(target=box_display, args=(child_pipe, width, height))
    box_process.start()
    print("[Box] Started.")

def stop_box():
    """Stop the box process."""
    global box_process, box_pipe
    if box_process and box_pipe:
        box_pipe.send("STOP")
        box_process.join()
        box_pipe = None
        print("[Box] Stopped.")
    else:
        print("[Box] Not running.")

def main():
    """Main function to run the prompt_toolkit shell."""
    session = PromptSession()
    print("Type 'start', 'stop', or 'exit' to control the box.")
    while True:
        try:
            command = session.prompt("> ").strip().lower()
            if command == "start":
                start_box(width=30, height=10)
            elif command == "stop":
                stop_box()
            elif command == "clock":
                create_clock_widget()
            elif command == "shell":
                render_shell("ls -al")
            elif command == "pop":
                render_in_panel("hello world")
            elif command == "exit":
                stop_box()
                print("Exiting...")
                break
            else:
                print("Unknown command. Type 'start', 'stop', or 'exit'.")
        except (EOFError, KeyboardInterrupt):
            stop_box()
            print("\nExiting...")
            break

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Use 'spawn' to ensure compatibility
    main()

