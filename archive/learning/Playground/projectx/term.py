#!/usr/bin/env python3
#
# term.py

import tkinter as tk
import subprocess

class SimpleTerminal(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        
        self.cmd_entry = tk.Entry(self, width=80)
        self.cmd_entry.bind("<Return>", self.run_command)
        self.cmd_entry.pack()

        self.output_text = tk.Text(self, width=80, height=24)
        self.output_text.pack()

    def run_command(self, event):
        command = self.cmd_entry.get()
        if command:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()

            self.output_text.insert(tk.END, f"\n> {command}\n")
            self.output_text.insert(tk.END, output.decode())
            if error:
                self.output_text.insert(tk.END, error.decode())

            self.cmd_entry.delete(0, tk.END)
            self.output_text.see(tk.END)  # Scroll the Text field to the end.

root = tk.Tk()
terminal = SimpleTerminal(master=root)
terminal.mainloop()

