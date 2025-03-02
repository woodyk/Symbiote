#!/usr/bin/env python3
#
# /tmp/tt.py

import customtkinter as ctk
import tkinter as tk
import subprocess
import threading
import os
import pty
import re

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class Terminal(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("1100x580")
        self.title("SymChat")
        self.font_size = 16

        # Calculate the width for the chat_history box
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        history_width = int(window_width * 0.6)  # Estimate 10 pixels per character
        history_height = int(window_height * 0.8)
        print(window_width, window_height)

        # Create a Text widget for output and input with a fixed-width font
        self.text = ctk.CTkTextbox(self, width=history_width, font=('Courier', self.font_size), state='normal')
        self.text.grid(row=0, column=1, rowspan=6, sticky='nsew')
        #self.text.pack(side='top', fill='both', expand=True, pady=(0, 5))

        # Create an Entry widget for input
        self.entry = ctk.CTkEntry(self, width=history_width, font=('Courier', self.font_size))
        self.entry.grid(row=6, column=1, sticky='nsew')

        # Bind Ctrl-Z to send_ctrl_z
        self.entry.bind('<Control-z>', self.send_ctrl_z)
        #self.entry.pack(side='bottom', fill='x', expand=True, pady=(5, 0))


        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Create two dropdowns
        self.dropdown1 = ctk.CTkOptionMenu(self.sidebar_frame, values=["Option 1", "Option 2", "Option 3"])
        self.dropdown1.grid(row=0, column=0, padx=20, pady=20)
        self.dropdown2 = ctk.CTkOptionMenu(self.sidebar_frame, values=["Option 1", "Option 2", "Option 3"])
        self.dropdown2.grid(row=1, column=0, padx=20, pady=20)

        # Create three buttons
        self.button1 = ctk.CTkButton(self.sidebar_frame, text="Button 1")
        self.button1.grid(row=2, column=0, padx=20, pady=20)
        self.button2 = ctk.CTkButton(self.sidebar_frame, text="Button 2")
        self.button2.grid(row=3, column=0, padx=20, pady=20)
        self.button3 = ctk.CTkButton(self.sidebar_frame, text="Button 3")
        self.button3.grid(row=4, column=0, padx=20, pady=20)

        # Configure the grid to expand properly
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5), weight=2)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        # Bind Return to send_input
        self.entry.bind('<Return>', self.send_input)

        # Start a subprocess
        master_fd, slave_fd = pty.openpty()
        self.process = subprocess.Popen(['symbiote', '-p'], stdin=slave_fd, stdout=slave_fd, stderr=slave_fd)
        self.master_fd = master_fd

        # Start a thread to read output from the subprocess
        threading.Thread(target=self.read_output).start()

    def send_input(self, event=None):
        # Get the input from the current line in the entry widget
        input_dat = self.entry.get().strip()
        print(f'sending {input_dat}')

        if input_dat == 'clear::':
            self.clear_text()
        else:
            # Send the input to the subprocess
            os.write(self.master_fd, (input_dat + '\x1b\n').encode('utf-8'))

        # Clear the Entry widget
        self.entry.delete(0, 'end')

        # Prevent the default behavior of the Return key
        return 'break'

    def read_output(self):
        # Regular expression to match ANSI escape sequences
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

        # Mapping from ANSI color escape sequences to Tkinter colors
        color_map = {
            '\x1B[0m': 'black',  # Reset
            '\x1B[31m': 'red',  # Red
            '\x1B[32m': 'green',  # Green
            '\x1B[33m': 'yellow',  # Yellow
            '\x1B[34m': 'blue',  # Blue
            '\x1B[35m': 'magenta',  # Magenta
            '\x1B[36m': 'cyan',  # Cyan
            '\x1B[37m': 'white',  # White
            # Add more colors as needed...
        }

        while True:
            # Read a line of output from the subprocess
            line = os.read(self.master_fd, 2048).decode('utf-8')

            # Check for CPR request
            if line == '\x1B[6n':
                # Calculate the current cursor position
                row, col = self.text.index('insert').split('.')
                row, col = int(row), int(col)

                # Send the cursor position to the subprocess
                os.write(self.master_fd, '\x1B[{};{}R'.format(row, col).encode('utf-8'))
            else:
                # Split the line into segments based on the ANSI escape sequences
                segments = ansi_escape.split(line)

                for segment in segments:
                    # If the segment is an ANSI escape sequence, get the corresponding color
                    if segment in color_map:
                        color = color_map[segment]
                    else:
                        color = 'green'
                        # Otherwise, insert the segment into the chat_history text box with the current color
                        segment = re.sub(r'\r\n', '\n', segment)
                        print(segment)
                        self.text.insert(tk.END, segment)
                        #self.text.tag_add(color, 'end - %d chars' % len(segment), tk.END)
                        self.text.tag_config(color, background=color)

                # Scroll the Text widget to the end
                self.text.yview(tk.END)

    def clear_text(self):
        self.text.delete('1.0', 'end')

    def send_ctrl_z(self, event=None):
        # Send a Ctrl-Z to the subprocess
        os.write(self.master_fd, bytes([26]))

        # Prevent the default behavior of Ctrl-Z
        return 'break'

if __name__ == "__main__":
    app = Terminal()
    app.mainloop()

#root = tk.Tk()
#root.geometry('1024x768')
#term = Terminal(root)
#term.pack(fill='both', expand=True)
#root.mainloop()
