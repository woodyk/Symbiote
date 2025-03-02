#!/usr/bin/env python3
#
# curses_panel.py

import curses
import curses.panel

def main(stdscr):
    curses.curs_set(0)  # Hide the cursor

    # Create two windows
    win1 = curses.newwin(10, 40, 5, 5)  # Height, width, start_y, start_x
    win2 = curses.newwin(10, 40, 8, 10)

    # Add borders and text
    win1.border()
    win1.addstr(1, 1, "Window 1: Panel 1", curses.A_BOLD)

    win2.border()
    win2.addstr(1, 1, "Window 2: Panel 2", curses.A_BOLD)

    # Create panels
    panel1 = curses.panel.new_panel(win1)
    panel2 = curses.panel.new_panel(win2)

    # Initially show panel 1 on top
    panel1.top_panel()

    # Update display
    curses.panel.update_panels()
    curses.doupdate()

    while True:
        key = stdscr.getch()
        if key == ord('q'):  # Quit
            break
        elif key == ord('1'):  # Show panel 1 on top
            panel1.top_panel()
        elif key == ord('2'):  # Show panel 2 on top
            panel2.top_panel()

        # Refresh panels
        curses.panel.update_panels()
        curses.doupdate()

if __name__ == "__main__":
    curses.wrapper(main)

