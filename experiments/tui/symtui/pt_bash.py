#!/usr/bin/env python3
#
# pt_bash.py
import os
import pty
import select
import threading
import time
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.widgets import Label
from prompt_toolkit.key_binding import KeyBindings

# Key bindings
kb = KeyBindings()
stop_rendering = threading.Event()

@kb.add('c-c')
def _handle_ctrl_c(event):
    sys.exit()

def render_panel(app):
    """Continuously render a panel in the top-right corner."""
    while not stop_rendering.is_set():
        # Update the UI content dynamically (e.g., timestamp)
        panel_label.text = f"Panel: {time.strftime('%H:%M:%S')}"
        app.invalidate()  # Redraw the application
        time.sleep(1)  # Update every second

@kb.add('c-b')
def _(event):
    """Launch Bash interactively within a pseudo-terminal."""
    global stop_rendering
    stop_rendering.clear()

    def spawn_bash():
        # Create a new pseudo-terminal
        master_fd, slave_fd = pty.openpty()

        # Fork the process
        pid = os.fork()
        if pid == 0:  # Child process
            os.close(master_fd)  # Close master FD in child
            os.dup2(slave_fd, 0)  # Standard input
            os.dup2(slave_fd, 1)  # Standard output
            os.dup2(slave_fd, 2)  # Standard error
            os.execvp("bash", ["bash", "--login", "-i"])  # Execute Bash
        else:  # Parent process
            os.close(slave_fd)  # Close slave FD in parent
            try:
                while True:
                    # Monitor input/output using select
                    r, _, _ = select.select([master_fd, 0], [], [])
                    if 0 in r:  # Input from user
                        try:
                            data = os.read(0, 1024)
                            os.write(master_fd, data)
                        except OSError:
                            break  # Exit on error (e.g., input closed)

                    if master_fd in r:  # Output from Bash
                        try:
                            data = os.read(master_fd, 1024)
                            if not data:
                                break  # Break if no more data (Bash exited)
                            os.write(1, data)
                        except OSError:
                            break  # Exit on error

            finally:
                os.close(master_fd)  # Close the pseudo-terminal
                os.waitpid(pid, 0)  # Wait for child process to terminate

    # Start the thread to render the panel
    thread = threading.Thread(target=render_panel, args=(event.app,), daemon=True)
    thread.start()

    spawn_bash()
    stop_rendering.set()  # Stop the panel thread
    print("Returned from Bash!")

# Define the top-right panel
panel_label = Label("Panel: Loading...", dont_extend_width=True)
panel = Window(content=panel_label, width=20, height=1)

# Define the layout with the panel in the top-right corner
layout = Layout(
    VSplit([
        HSplit([]),  # Empty placeholder for Bash rendering
        panel,       # The panel in the top-right corner
    ])
)

# Create the application
app = Application(
    layout=layout,
    full_screen=True,
    key_bindings=kb,
)

if __name__ == "__main__":
    print("Press Ctrl-B to launch Bash shell.")
    app.run()

