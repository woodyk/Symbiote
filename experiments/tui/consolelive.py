#!/usr/bin/env python3
#
# consolelive.py

import os
import subprocess
import time
import uuid
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from threading import Thread
from queue import Queue

# Box manager to handle active boxes
class BoxManager:
    def __init__(self):
        self.boxes = {}  # {id: {"command": str, "output": str, "width": int, "height": int}}
        self.console = Console()

    def add_box(self, command):
        box_id = str(uuid.uuid4())[:8]
        self.boxes[box_id] = {
            "command": command,
            "output": "",
            "width": 30,
            "height": 10,
        }
        # Start a thread to update the box's output
        Thread(target=self._update_box_output, args=(box_id,), daemon=True).start()
        return box_id

    def remove_box(self, box_id):
        if box_id in self.boxes:
            del self.boxes[box_id]
            return True
        return False

    def resize_box(self, box_id, direction):
        if box_id in self.boxes:
            if direction == "larger":
                self.boxes[box_id]["width"] += 10
                self.boxes[box_id]["height"] += 5
            elif direction == "smaller":
                self.boxes[box_id]["width"] = max(10, self.boxes[box_id]["width"] - 10)
                self.boxes[box_id]["height"] = max(5, self.boxes[box_id]["height"] - 5)

    def get_box_ids(self):
        return list(self.boxes.keys())

    def _update_box_output(self, box_id):
        command = self.boxes[box_id]["command"]
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while process.poll() is None:
            output = process.stdout.readline().decode()
            if output:
                self.boxes[box_id]["output"] += output

    def render(self):
        layout = Layout()
        layout.split_column()
        for box_id, box_data in self.boxes.items():
            box_panel = Panel(
                box_data["output"][-500:],  # Display the last 500 characters of output
                title=f"Box {box_id}",
                width=box_data["width"],
                height=box_data["height"],
            )
            layout.split_row(*[Panel(box_panel, title=box_id)])
        return layout

def handle_input(manager, command):
    """Process commands entered by the user."""
    if command.startswith("box:start:"):
        cmd = command[10:].strip("\\")
        box_id = manager.add_box(cmd)
        manager.console.print(f"[green]Box started with ID {box_id}.[/green]")
    elif command.startswith("box:stop:"):
        box_id = command[9:].strip()
        if manager.remove_box(box_id):
            manager.console.print(f"[red]Box {box_id} stopped.[/red]")
        else:
            manager.console.print(f"[red]Box {box_id} not found.[/red]")
    elif command == "box:ids":
        ids = manager.get_box_ids()
        manager.console.print(f"[cyan]Active boxes: {', '.join(ids)}[/cyan]")
    elif command.startswith("box:larger:"):
        box_id = command[11:].strip()
        manager.resize_box(box_id, "larger")
        manager.console.print(f"[blue]Box {box_id} resized larger.[/blue]")
    elif command.startswith("box:smaller:"):
        box_id = command[12:].strip()
        manager.resize_box(box_id, "smaller")
        manager.console.print(f"[blue]Box {box_id} resized smaller.[/blue]")
    else:
        # Execute shell command
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout or result.stderr
            manager.console.print(f"[white]{output}[/white]")
        except Exception as e:
            manager.console.print(f"[red]Error executing command: {e}[/red]")

def main():
    console = Console()
    manager = BoxManager()

    # Queue for user input
    input_queue = Queue()

    # Function to capture user input
    def input_thread():
        while True:
            command = input("> ").strip()
            input_queue.put(command)

    

    # Start input thread
    Thread(target=input_thread, daemon=True).start()


    with Live(console=console, auto_refresh=False) as live:
        while True:
            if not input_queue.empty():
                command = input_queue.get()
                handle_input(manager, command)
            time.sleep(0.1)

if __name__ == "__main__":
    main()

