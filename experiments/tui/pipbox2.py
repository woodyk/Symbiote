#!/usr/bin/env python3

import os
import multiprocessing
import signal
import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from prompt_toolkit import PromptSession
console = Console()
print = console.print
log = console.log

# Global variables for inter-process communication
box_process = None
box_pipe = None

def box_display(pipe, width, height):
    """
    The process to handle the live-refreshing box display.
    Args:
        pipe: The multiprocessing pipe for communication.
        width: Width of the box.
        height: Height of the box.
    """
    try:
        while True:
            if pipe.poll():
                message = pipe.recv()
                if message == "STOP":
                    break
            # Update content dynamically
            content = Text(f"Live Data:\n{time.strftime('%H:%M:%S')}", justify="center")
            live.update(Panel(content, width=width, height=height, title="Info Box"))
            time.sleep(0.1)  # Simulate refresh delay
    except KeyboardInterrupt:
        pass

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
    with Live(console=console, auto_refresh=False) as live:
        while True:
            try:
                command = session.prompt("> ").strip().lower()
                if command == "start":
                    start_box(width=30, height=10)
                elif command == "stop":
                    stop_box()
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
