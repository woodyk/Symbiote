#!/usr/bin/env python3
#
# prompt_example.py

import os
from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import VSplit, FloatContainer, HSplit, Window
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import SearchToolbar, TextArea, Frame
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.layout.dimension import Dimension

def chatBot():
    terminal_width, terminal_height = os.get_terminal_size()

    # Calculate desired dimensions as a percentage
    width_percent = 0.4  # 60% of terminal width
    height_percent = 0.4  # 40% of terminal height
    calculated_width = int(terminal_width * width_percent)
    calculated_height = int(terminal_height * height_percent)

    style = Style.from_dict({
        "frame.border": "fg:white",
        "frame.title": "fg:cyan",
        #"content": "fg:white bg:#333333",
        "content": "fg:white",
        #"output-field": "bg:#000044 #ffffff",
        "output-field": "#ffffff",
        "input-field": "bg:#000000 #ffffff",
        "line": "#004400"
    })

    search_field = SearchToolbar()
    
    output_field = TextArea("hello", style="class:output-field", scrollbar=True)
    input_field = TextArea(
            height=10,
            prompt=">>> ",
            style="class:input-field",
            multiline=True,
            wrap_lines=False,
            search_field=search_field
    )

    # The key bindings.
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event):
        event.app.exit()

    hsplit_content = HSplit([
        output_field,
        Window(height=1, style="class:line"),
        input_field,
        search_field,
    ])

    framed_hsplit = Frame(
            hsplit_content,
            title="my application",
            style="class:frame",
            height=calculated_height
        )

    def responder(text):
        response = f"Ai:\nYou asked me. --> {text}\n\n"
        return response
    
    def accept(buff):
        try:
            output = f"User:\n{input_field.text}\n\n"
        except BaseException as e:
            output = f"\n\n{e}"

        output += responder(input_field.text)

        # Append our input to the output
        new_text = output_field.text + output

        # Add text to output buffer.
        output_field.buffer.document = Document(
            text=new_text, cursor_position=len(new_text)
        )

    # Captures when there is an enter at the input
    input_field.accept_handler = accept

    layout = Layout(framed_hsplit, focused_element=input_field)

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        mouse_support=True,
        full_screen=True,
    )

    app.run()

def prompt_session_container():
    style = Style.from_dict({
        "root": "bg:#1e1e1e fg:white",  # Set the whole background to a dark gray
        "prompt": "bg:#333333 fg:#00ff00",  # Style the prompt text area
        "input": "bg:#1e1e1e fg:white",  # Style the input text area
        "frame.border": "red",
    })

    session = PromptSession("Enter your command: ", style=style)

    session.prompt()


if __name__ == "__main__":
    chatBot()
    session = PromptSession()
    #prompt_session_container()
