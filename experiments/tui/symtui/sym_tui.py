#!/usr/bin/env python3
#
# sym_tui.py

import os
import pty
import tty
import select

def run_bash():
    # Open a pseudo-terminal
    master_fd, slave_fd = pty.openpty()

    # Fork the process
    pid = os.fork()

    if pid == 0:  # Child process
        # In the child, replace with bash shell
        os.close(master_fd)  # Close the master file descriptor

        # Set up the slave PTY as the standard input/output/error
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)  # Close the slave descriptor

        # Execute bash
        os.execlp("bash", "bash")
    else:  # Parent process
        os.close(slave_fd)  # Close the slave descriptor in the parent

        # Save original terminal settings
        old_tty_attrs = tty.tcgetattr(0)
        tty.setraw(0)  # Set the terminal to raw mode

        try:
            # Relay input/output between the user and bash
            while True:
                rlist, _, _ = select.select([master_fd, 0], [], [])
                for fd in rlist:
                    if fd == master_fd:  # Output from bash
                        data = os.read(master_fd, 1024)
                        if data:
                            os.write(1, data)  # Write to stdout
                    else:  # Input from user
                        data = os.read(0, 1024)
                        if data:
                            os.write(master_fd, data)  # Send to bash
        except OSError:
            pass  # Exit gracefully on error
        finally:
            # Restore original terminal settings
            tty.tcsetattr(0, tty.TCSADRAIN, old_tty_attrs)
            os.waitpid(pid, 0)  # Wait for the child process to exit

if __name__ == "__main__":
    run_bash()

