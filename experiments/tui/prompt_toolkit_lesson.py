#!/usr/bin/env python3
#
# prompt_toolkit_lesson.py

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.widgets import TextArea, Frame
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.buffer import Buffer

# Key bindings
kb = KeyBindings()

@kb.add("c-c")
def exit_app(event):
    event.app.exit()

# Dynamic content for the top area
log_window = TextArea(text="Logs will appear here...\n", scrollbar=True)

# Prompt locked at the bottom
prompt_buffer = Buffer()
prompt_window = Frame(
    title="Input",
    body=Window(content=BufferControl(buffer=prompt_buffer)),
)

# Application layout
root_container = HSplit([
    log_window,
    prompt_window,
])

layout = Layout(root_container)

# Fullscreen application
app = Application(
    layout=layout,
    full_screen=True,
    key_bindings=kb,
)

if __name__ == "__main__":
    app.run()

