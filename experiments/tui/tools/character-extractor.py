#!/usr/bin/env python3
#
# character-extractor.py

import os
import re
import sys

def extract_unique_characters_from_file(file_path):
    """Extract unique characters from a single file, handling binary files."""
    unique_characters = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                unique_characters.update(line)
    except (UnicodeDecodeError, IOError):
        # Handle binary files or unreadable files quietly
        unique_characters.update(extract_strings_from_binary(file_path))
    return unique_characters

def extract_strings_from_binary(file_path, min_length=4):
    """Extract readable strings from binary files."""
    unique_characters = set()
    try:
        with open(file_path, 'rb') as file:
            data = file.read()
            # Regex to extract readable strings (ASCII or UTF-8)
            strings = re.findall(b'[\x20-\x7E]{%d,}' % min_length, data)
            for s in strings:
                unique_characters.update(s.decode('utf-8', errors='ignore'))
    except Exception:
        pass  # Suppress all errors
    return unique_characters

def extract_unique_characters_from_directory(directory_path):
    """Extract unique characters from all files in a directory."""
    unique_characters = set()
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            unique_characters.update(extract_unique_characters_from_file(file_path))
    return unique_characters

def extract_unique_characters_from_stdin():
    """Extract unique characters from stdin input."""
    unique_characters = set()
    for line in sys.stdin:
        unique_characters.update(line)
    return unique_characters

def print_table(characters, columns=30, spacing=2):
    """Print unique characters in a table format."""
    characters = sorted(characters)
    for i, char in enumerate(characters):
        print(char.ljust(spacing), end="")
        if (i + 1) % columns == 0:
            print()  # Move to a new line after `columns` characters
    if len(characters) % columns != 0:
        print()  # Final newline if the last row is incomplete

def main():
    if len(sys.argv) == 1 and not sys.stdin.isatty():
        # Handle case where input is piped (e.g., `cat file.txt | charextractor.py`)
        unique_characters = extract_unique_characters_from_stdin()
    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        if os.path.isfile(input_path):
            unique_characters = extract_unique_characters_from_file(input_path)
        elif os.path.isdir(input_path):
            unique_characters = extract_unique_characters_from_directory(input_path)
        else:
            print("Error: Invalid path. Please provide a valid file or directory.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage:")
        print("  charextractor.py <file>       # Extract unique characters from a file")
        print("  charextractor.py <directory> # Extract unique characters from all files in a directory")
        print("  cat <file> | charextractor.py # Extract unique characters from stdin")
        sys.exit(1)

    print("\nUnique Characters Found:\n")
    print_table(unique_characters, columns=30, spacing=2)

if __name__ == "__main__":
    main()

