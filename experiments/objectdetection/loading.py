#!/usr/bin/env python3
#
# loading.py

import time
from halo import Halo

# Create a Halo spinner instance
spinner = Halo(text='Loading', spinner='dots')

# Start the spinner
spinner.start()

# Simulate some loading time with a sleep function
time.sleep(3)

# Stop the spinner and display a success message with a checkmark
spinner.succeed('Completed!')

