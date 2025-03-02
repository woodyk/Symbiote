#!/usr/bin/env python3
#
# main.py

import openai
import re
import threading
from pynput.keyboard import Listener
import tkinter as tk
from tkinter import Toplevel, Text, Entry, END, Scrollbar, N, S, E, W

class ChatWindow:
    def __init__(self, master, initial_message):
        self.window = Toplevel(master)
        self.window.title('Chat with GPT')
        self.window.attributes('-topmost', True)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)  # register callback function when window is deleted
        self.text_area = Text(self.window)
        self.text_area.insert('1.0', initial_message)
        self.text_area['state'] = 'disabled'
        self.text_area.pack()

        self.entry_field = Entry(self.window)
        self.entry_field.bind('<Return>', self.on_enter_pressed)
        self.entry_field.pack()

    def on_enter_pressed(self, event):
        message = self.entry_field.get()
        self.entry_field.delete(0, 'end')
        self.add_message("You: " + message)

        response = self.generate_response(message)  # replace with actual call to GPT-3
        self.add_message("GPT: " + response)

    def add_message(self, message):
        self.text_area['state'] = 'normal'
        self.text_area.insert('end', '\n' + message)
        self.text_area['state'] = 'disabled'

    def generate_response(self, message):
        return "You said: " + message  # replace this with actual call to GPT-3

    def on_close(self):
        keylogger.begin() 
        self.window.destroy()

class KeyLogger:
    def __init__(self):
        self.log = ""
        self.chat_is_active = False
        self.window = tk.Tk()
        self.window.withdraw()  # hide main window
        self.chat_window = None

    def on_press(self, key):
        if self.chat_is_active:
            return

        try:
            pressed = re.sub('\'', "", str(key))

            if not re.search("Key\.*", pressed):
                self.log += pressed 
            
            print(pressed)

            if re.search(":help::", self.log):
                print("Help menu triggered")
                self.chat_is_active = True
                self.show_chat_window(self.log)
                self.log = ""  # reset the log

        except AttributeError:
            print("Unable to capture key.")

    def show_chat_window(self, message):
        if self.chat_window is not None:
            self.chat_window.window.destroy()

        self.chat_window = ChatWindow(self.window, message)

    def start(self):
        listener = Listener(on_press=self.on_press)
        self.t = threading.Thread(target=listener.start)
        self.t.start()

        self.window.mainloop()

    def begin(self):
        self.chat_is_active = False 

if __name__ == "__main__":
    global keylogger
    keylogger = KeyLogger()
    keylogger.start()

