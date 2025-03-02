#!/usr/bin/env python3
#
# symbiote/symgui.py

import os
import sys
import re
import tempfile
import subprocess
import threading
import tkinter as tk
import tkinter.messagebox
import customtkinter as ctk

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class ChatApp(ctk.CTk):
    class IORedirector(object):
        def __init__(self, text_area):
            self.text_area = text_area

        def write(self, str):
            self.text_area.insert(tk.END, str)
            self.flush()

        def flush(self):
            # Update the display of the text area
            self.text_area.update_idletasks()

        def fileno(self):
            return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry(f"{1100}x{580}")
        self.title("SymChat")
        self.font_size = 20

        # configure grid layout (x4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Symbiote", font=ctk.CTkFont(size=self.font_size, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = ctk.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = ctk.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))


        # Calculate the width for the chat_history box
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        chat_history_width = int(window_width * 0.6)  # Estimate 10 pixels per character
        chat_history_height = int(window_height * 0.8)

        # create chat frame
        self.chat_frame = ctk.CTkFrame(self, width=chat_history_width) 
        self.chat_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")

        # Create a text box for chat history
        self.chat_history = ctk.CTkTextbox(self.chat_frame, width=chat_history_width, height=chat_history_height)
        self.chat_history.grid(row=0, column=1, rowspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Start the symbiote CLI in a subprocess
        self.process = subprocess.Popen(['symbiote'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

        # Start a thread to read output from the subprocess and display it in the chat_history text box
        threading.Thread(target=self.read_output).start()

        # Redirect stdout
        sys.stdout = self.IORedirector(self.chat_history)

        # Create an entry field for typing messages
        '''
        self.message_entry = ctk.CTkEntry(self)
        self.message_entry.pack(fill=tk.X, expand=True, padx=20, pady=20, anchor='e')
        '''

        # create main entry and button
        self.message_entry = ctk.CTkEntry(self.chat_frame, placeholder_text="", width=chat_history_width, height=2)
        self.message_entry.grid(row=4, column=1, padx=(20, 20), pady=(0, 0), sticky="nsew")

        # Bind Return to send_message and Shift+Return to insert newline
        self.message_entry.bind('<Return>', self.send_message)
        self.message_entry.bind('<Shift-Return>', lambda event: self.message_entry.insert(tk.END, '\n'))

        # Load symbiote chat
        current_path = os.getcwd()
        self.schat = chat.symchat(working_directory=current_path, debug=False)

        # create radiobutton frame
        self.radiobutton_frame = ctk.CTkFrame(self)
        self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.radio_var = tkinter.IntVar(value=0)
        self.label_radio_group = ctk.CTkLabel(master=self.radiobutton_frame, text="CTkRadioButton Group:")
        self.label_radio_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=0)
        self.radio_button_1.grid(row=1, column=2, pady=10, padx=20, sticky="n")
        self.radio_button_2 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=1)
        self.radio_button_2.grid(row=2, column=2, pady=10, padx=20, sticky="n")
        self.radio_button_3 = ctk.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=2)
        self.radio_button_3.grid(row=3, column=2, pady=10, padx=20, sticky="n")

        # create checkbox and switch frame
        self.checkbox_slider_frame = ctk.CTkFrame(self)
        self.checkbox_slider_frame.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.checkbox_1 = ctk.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_1.grid(row=1, column=0, pady=(20, 0), padx=20, sticky="n")
        self.checkbox_2 = ctk.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_2.grid(row=2, column=0, pady=(20, 0), padx=20, sticky="n")
        self.checkbox_3 = ctk.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_3.grid(row=3, column=0, pady=20, padx=20, sticky="n")

        # set default values
        self.sidebar_button_3.configure(state="disabled", text="Disabled CTkButton")
        self.checkbox_3.configure(state="disabled")
        self.checkbox_1.select()
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

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

        for line in iter(self.process.stdout.readline, ''):
            # Split the line into segments based on the ANSI escape sequences
            segments = ansi_escape.split(line)

            for segment in segments:
                # If the segment is an ANSI escape sequence, get the corresponding color
                if segment in color_map:
                    color = color_map[segment]
                else:
                    color = 'green'
                    # Otherwise, insert the segment into the chat_history text box with the current color
                    self.chat_history.insert(tk.END, segment)
                    self.chat_history.tag_add(color, 'end - %d chars' % len(segment), tk.END)
                    self.chat_history.tag_config(color, foreground=color)

            self.chat_history.see(tk.END)



    def open_input_dialog_event(self):
        dialog = ctk.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

    '''
    def send_message(self, event=None):
        # Get the message from the entry field
        message = self.message_entry.get()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name

        # Redirect stdout to the temporary file
        sys.stdout = open(temp_path, 'w')

        # Send the message using symchat
        response = self.schat.chat(user_input=message, run=True)  # Replace with your actual function
        if isinstance(response, list):
            chat_response = response[0].pop()
            returned = chat_response['content']

        # Add the message, response, and output to the chat history
        self.chat_history.insert(tk.END, "symchat> " + message + "\n")

        # Close and reopen the temporary file in read mode
        sys.stdout.close()
        sys.stdout = open(temp_path, 'r')

        # Read the contents of the temporary file
        output = sys.stdout.read()

        # Redirect stdout back to the text box
        sys.stdout.close()
        sys.stdout = self.IORedirector(self.chat_history)

        self.chat_history.insert(tk.END, output)

        # Scroll to the bottom of the chat history
        self.chat_history.see(tk.END)

        # Clear the entry field
        self.message_entry.delete(0, tk.END)

        # Delete the temporary file
        os.remove(temp_path)
    '''
    def send_message(self, event=None):
        # Get the message from the entry field
        message = self.message_entry.get()

        # Send the message to the subprocess
        self.process.stdin.write(message + '\n')
        self.process.stdin.flush()

        # Clear the entry field
        self.message_entry.delete(0, tk.END)

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()

