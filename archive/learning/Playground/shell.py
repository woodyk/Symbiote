#!/usr/bin/env python3
#
# shell.py

import subprocess
import os
import sys
import select
import termios
import tty
import re

def main():
    command = ['bash', '--norc', '--noprofile']
    pty_master, pty_slave = os.openpty()
    subprocess.Popen(command, stdin=pty_slave, stdout=pty_slave, stderr=pty_slave, start_new_session=True)

    # Save the terminal settings
    old_settings = termios.tcgetattr(sys.stdin)

    # Set the terminal to raw mode
    tty.setraw(sys.stdin)

    command_buffer = ""
    response_buffer = ""
    session_data = []
    response = ""
    command = ""
    while True:
        r, _, _ = select.select([sys.stdin, pty_master], [], [])

        if sys.stdin in r:
            # Read input from the user
            data = os.read(sys.stdin.fileno(), 1024)
            command_buffer += data.decode('utf-8')

            if 'exit::\r' in command_buffer:
            #if re.search(r'exit::\r', command_buffer):
                # Execute your own code
                print("Exiting...")
                break

            os.write(pty_master, data)

        if pty_master in r:
            # Read output from the subprocess
            response_data = os.read(pty_master, 1024)
            response_buffer += response_data.decode('utf-8').strip()
           
            if re.search(r'\S+\$', response_buffer):
                response_buffer = re.sub(r'\S+\$', "", response_buffer)
                command = response_buffer

            if '\r\n' in response_buffer:
                # Strip out command prompt from response buffer
                response = re.sub(r'\S+\$|\x1b\[\?2004l', "", response_buffer)

                session_data.append({ "command": command, "response": response })
                response_buffer = ""

            os.write(sys.stdout.fileno(), response_data)

            # Check if the subprocess has exited
            if not response_data:
                break

    print(session_data)
    # Restore the terminal settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == '__main__':
    main()
