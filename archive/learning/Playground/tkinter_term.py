#!/usr/bin/env python3
#
# tt.py

import re
import tkinter as tk
import subprocess
import threading

class Terminal(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        # Create a Text widget for output and input
        self.text = tk.Text(self)
        self.text.pack(side='top', fill='both', expand=True)

        # Bind Return to send_input
        self.text.bind('<Return>', self.send_input)

        # Start a subprocess
        self.process = subprocess.Popen(['symbiote', '-p'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Start a thread to read output from the subprocess
        threading.Thread(target=self.read_output).start()

    def send_input(self, event=None):
        # Get the input from the current line in the Text widget
        input = self.text.get('insert linestart', 'insert lineend')
        print(f'sending {input}')

        # Send the input to the subprocess
        self.process.stdin.write(input.encode('utf-8'))
        self.process.stdin.flush()

        # Send an ESC-Enter combination
        self.process.stdin.write('\x1b\n'.encode('utf-8'))
        self.process.stdin.flush()


        # Insert a newline at the end of the Text widget
        self.text.insert('end', '\n')

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
            for line in iter(self.process.stdout.readline, ''):
                if re.match(r'^\r\n', str(line)):
                    continue

                if re.match(r'^symchat> ', str(line)):
                    break

                # Split the line into segments based on the ANSI escape sequences
                segments = ansi_escape.split(str(line))

                self.text.see(tk.END)

                # Insert the line into the Text widget
                self.text.insert('end', line.decode('utf-8'))

                # Scroll the Text widget to the end
                self.text.yview('end')


            if not line:
                break

            '''

            for segment in segments:
                # If the segment is an ANSI escape sequence, get the corresponding color
                if segment in color_map:
                    color = color_map[segment]
                else:
                    color = 'green'
                    # Otherwise, insert the segment into the chat_history text box with the current color
                    self.text.insert(tk.END, segment)
                    self.text.tag_add(color, 'end - %d chars' % len(segment), tk.END)
                    self.text.tag_config(color, foreground=color)

            self.text.see(tk.END)

            # Insert the line into the Text widget
            self.text.insert('end', line.decode('utf-8'))

            # Scroll the Text widget to the end
            self.text.yview('end')
            '''

root = tk.Tk()
term = Terminal(root)
term.pack(fill='both', expand=True)
root.mainloop()
