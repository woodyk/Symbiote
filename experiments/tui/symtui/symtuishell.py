#!/usr/bin/env python3
#
# symtuishell.py

import os
import pty
import select
import sys
import termios
import tty

def launch_shell():
    # Open a pseudo-terminal
    master_fd, slave_fd = pty.openpty()

    pid = os.fork()
    if pid == 0:  # Child process
        os.close(master_fd)  # Close master in child
        os.dup2(slave_fd, 0)  # stdin
        os.dup2(slave_fd, 1)  # stdout
        os.dup2(slave_fd, 2)  # stderr
        os.environ["TERM"] = "xterm-256color"  # Ensure TERM is set for proper behavior
        os.close(slave_fd)
        os.execlp("bash", "bash")  # Replace with bash
    else:  # Parent process
        os.close(slave_fd)  # Close slave in parent

        # Save original terminal settings
        old_tty_attrs = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin)  # Set terminal to raw mode

        try:
            while True:
                rlist, _, _ = select.select([master_fd, sys.stdin], [], [])
                for fd in rlist:
                    if fd == master_fd:  # Output from bash
                        data = os.read(master_fd, 1024)
                        if data:
                            os.write(sys.stdout.fileno(), data)
                    elif fd == sys.stdin.fileno():  # Input from user
                        data = os.read(sys.stdin.fileno(), 1024)
                        if data:
                            os.write(master_fd, data)
        except OSError:
            pass  # Handle exit gracefully
        finally:
            # Restore original terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty_attrs)
            os.waitpid(pid, 0)  # Wait for child process to exit

if __name__ == "__main__":
    launch_shell()
