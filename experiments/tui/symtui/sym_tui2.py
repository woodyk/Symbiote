#!/usr/bin/env python3
#
# sym_tui2.py

import os
import pty
import tty
import select
import curses

def run_bash_session(master_fd, slave_fd):
    """Sets up and launches the Bash session."""
    os.close(master_fd)  # Close the master in the child process
    os.setsid()  # Create a new session
    tty.setraw(slave_fd)  # Set the slave terminal to raw mode
    os.dup2(slave_fd, 0)  # Redirect stdin
    os.dup2(slave_fd, 1)  # Redirect stdout
    os.dup2(slave_fd, 2)  # Redirect stderr
    os.close(slave_fd)  # Close the slave after duplication
    os.environ["TERM"] = "xterm-256color"  # Set terminal type
    os.execlp("bash", "bash", "--noediting")  # Execute bash

def tui_overlay(screen, master_fd):
    """Curses overlay to render widgets and bash session output."""
    curses.curs_set(0)  # Hide cursor
    screen.nodelay(True)  # Make getch() non-blocking
    screen.clear()

    while True:
        # Handle bash output
        rlist, _, _ = select.select([master_fd], [], [], 0.1)
        if master_fd in rlist:
            data = os.read(master_fd, 1024).decode(errors='ignore')
            screen.addstr(1, 1, data)
            screen.refresh()

        # Check for user input
        key = screen.getch()
        if key == ord('q'):  # Quit on 'q'
            break

def run_bash_with_tui():
    """Runs the bash shell with TUI integration."""
    master_fd, slave_fd = pty.openpty()  # Open pseudo-terminal
    pid = os.fork()  # Fork the process

    if pid == 0:  # Child process
        run_bash_session(master_fd, slave_fd)
    else:  # Parent process
        os.close(slave_fd)  # Close slave FD in the parent
        curses.wrapper(tui_overlay, master_fd)  # Launch curses
        os.waitpid(pid, 0)  # Wait for child process to exit

if __name__ == "__main__":
    run_bash_with_tui()

