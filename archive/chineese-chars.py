#!/usr/bin/env python3
#
# chineese-chars.py

import sys

# Define the range for Chinese characters (CJK Unified Ideographs)
start = 0x4E00
end = 0x9FFF

# Collect all Chinese characters into a list
characters = [chr(i) for i in range(start, end + 1)]

# Define the number of columns
columns = 30 

# Print characters in columns
for i, char in enumerate(characters):
    if (i + 1) % columns == 0:
        print(char)  # Print character and move to the next line
    else:
        print(char, end=" ")  # Print character with a space

