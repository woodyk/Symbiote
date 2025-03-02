#!/usr/bin/env python3
#
# basic_tui.py

import os
import time
import sys

def draw_interface():
    os.system("clear")
    print("Logs or information above the prompt")
    print("-" * 80)
    print("Prompt locked here:")

def main():
    while True:
        draw_interface()
        try:
            user_input = input(">")
            if user_input.lower() == "exit":
                break
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

if __name__ == "__main__":
    main()

