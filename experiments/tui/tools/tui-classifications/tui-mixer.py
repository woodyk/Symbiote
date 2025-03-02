#!/usr/bin/env python3
#
# tui-mixer.py

import os
import argparse
from itertools import permutations

# Predefined patterns for TUI
TUI_PATTERNS = [
    "{0}━{1}━{2}\n{3}   {3}\n{4}━{5}━{6}",
    "{0}═{1}═{2}\n{3}   {3}\n{4}═{5}═{6}",
    "{0} {1} {2}\n{3}   {3}\n{4} {5} {6}",
    "{0}{1}{1}{1}{2}\n{3}       {3}\n{4}{5}{5}{5}{6}"
]

def generate_mixed_templates(characters, width=3):
    """
    Generate templates by mixing given characters into predefined patterns.
    :param characters: A string of characters to mix into patterns.
    :param width: Width of horizontal sections for consistent alignment.
    :return: A list of generated templates.
    """
    results = []
    for perm in permutations(characters, 7):  # Use 7 characters per pattern
        for pattern in TUI_PATTERNS:
            try:
                # Adjust template for fixed width
                horizontal = "━" * width
                template = pattern.format(
                    perm[0], horizontal, perm[1],
                    perm[2], horizontal, perm[3], perm[4]
                )
                results.append(template)
            except Exception:
                continue
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze fonts for TUI compatibility.")
    parser.add_argument("--mix", type=str, help="Mix a string of characters into TUI patterns.")
    args = parser.parse_args()

    if args.mix:
        print(f"\nMixing characters: {args.mix}\n")
        templates = generate_mixed_templates(args.mix, width=3)  # Fixed width for alignment
        print("\nGenerated Templates:")
        for template in templates[:10]:  # Limit output for readability
            print(template)
            print()
        return

    # If no arguments are provided, print usage
    parser.print_usage()

if __name__ == "__main__":
    main()

