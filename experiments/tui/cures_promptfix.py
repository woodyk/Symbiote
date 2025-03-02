#!/usr/bin/env python3
#
# cures_promptfix.py

import curses

def main(stdscr):
    curses.curs_set(1)  # Enable the cursor
    stdscr.clear()
    
    # Get screen size
    height, width = stdscr.getmaxyx()
    
    # Create a separate window for the prompt
    prompt_win = curses.newwin(1, width, height - 1, 0)  # 1-row high, full width, at the bottom
    
    # Log window above the prompt
    log_win = curses.newwin(height - 1, width, 0, 0)
    
    logs = ["Welcome to the TUI", "Logs will appear here..."]
    
    while True:
        # Render logs
        log_win.clear()
        for idx, log in enumerate(logs[-(height - 2):]):  # Show last few logs
            log_win.addstr(idx, 0, log)
        log_win.refresh()
        
        # Prompt for input
        prompt_win.clear()
        prompt_win.addstr(0, 0, "Your input: ")
        prompt_win.refresh()
        input_str = prompt_win.getstr(0, 12).decode("utf-8")  # Capture input
        
        # Add the input to logs or handle commands
        if input_str.lower() == "exit":
            break
        logs.append(f"You typed: {input_str}")

if __name__ == "__main__":
    curses.wrapper(main)

