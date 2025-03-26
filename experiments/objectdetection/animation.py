#!/usr/bin/env python3
#
# animation.py

import sys
import time
import threading

# Declare stop_event and animation_thread at a global level
stop_event = None
animation_thread = None

def launch_animation(state):
    global stop_event, animation_thread  # Access the global variables

    def hide_cursor():
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    def show_cursor():
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    def terminal_animation(stop_event):
        # Define static center character and rotating frames around it
        center_char = "\033[93m*\033[0m"  # Static center with yellow color

        # New character set pattern (blocks, shapes, and symbols)
        rotating_chars = ['▒', '░', '▓', '█', '◆', '◇', '▣', '▢', '○', '●', '◼', '◻']
        positions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # 8 positions around the center

        def move_cursor(x, y):
            """Moves the cursor to a relative position (x, y)."""
            sys.stdout.write(f"\033[{y+10};{x+10}H")  # Shift the position to the center of the terminal
            sys.stdout.flush()

        # Hide the cursor
        hide_cursor()

        # Start the animation loop
        frame = 0
        while not stop_event.is_set():
            # Move cursor to center and print the center character
            move_cursor(0, 0)
            sys.stdout.write(center_char)

            # Print rotating characters around the center in predefined positions
            for i, (dx, dy) in enumerate(positions):
                char = rotating_chars[(frame + i) % len(rotating_chars)]
                move_cursor(dx, dy)
                sys.stdout.write(f"\033[96m{char}\033[0m")  # Cyan-colored rotating chars

            sys.stdout.flush()
            time.sleep(0.15)

            # Return cursor to the same position and overwrite characters by clearing the region
            for i in range(-1, 2):
                for j in range(-1, 2):
                    move_cursor(i, j)
                    sys.stdout.write(" ")  # Overwrite with space to clear

            frame += 1

        # Restore the cursor and clear the animation
        sys.stdout.write("\033[2J")
        show_cursor()

    # Start or stop the animation based on the state
    if state is True:
        # Create an Event object to signal the thread to stop
        stop_event = threading.Event()

        # Start the animation thread
        animation_thread = threading.Thread(target=terminal_animation, args=(stop_event,))
        animation_thread.start()
    else:
        # Stop the animation by setting the stop_event
        stop_event.set()

        # Wait for the animation thread to finish
        animation_thread.join()

# Run the animation for 5 seconds
launch_animation(True)
time.sleep(5)
launch_animation(False)

