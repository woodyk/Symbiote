#!/usr/bin/env bash
#
# terminal_test.sh

#!/bin/bash

# Function to check ANSI color support
test_ansi_colors() {
    echo "Testing ANSI Colors:"
    for fg in {30..37}; do
        for bg in {40..47}; do
            echo -ne "\033[${fg};${bg}m ${fg};${bg} \033[0m "
        done
        echo
    done
    echo
}

# Function to test block elements rendering
test_block_elements() {
    echo "Testing Block Elements:"
    echo "Full Block: █"
    echo "Upper Half: ▀"
    echo "Lower Half: ▄"
    echo
}

# Function to test Braille patterns rendering
test_braille_patterns() {
    echo "Testing Braille Patterns:"
    for i in {0x2800..0x2808}; do
        printf "%b " "$(printf '\\U%04X' "$i")"
    done
    echo
    echo
}

# Function to test cursor positioning
test_cursor_positioning() {
    echo "Testing Cursor Positioning:"
    echo -ne "\033[10;10HHello at (10, 10)\n"
    echo -ne "\033[15;15HHello at (15, 15)\n"
    echo -ne "\033[20;20HHello at (20, 20)\n"
    echo
}

# Function to test sixel support
test_sixel_graphics() {
    echo "Testing Sixel Graphics (if supported):"
    echo -ne "\033Pq"  # Begin sixel
    echo -ne "#0;2;0;0;0;255~"  # A single blue pixel
    echo -ne "\033\\"  # End sixel
    echo
}

# Function to test TrueColor support
test_truecolor() {
    echo "Testing TrueColor (24-bit) Support:"
    for r in {0..5}; do
        for g in {0..5}; do
            for b in {0..5}; do
                color=$((16 + r * 36 + g * 6 + b))
                echo -ne "\033[38;5;${color}m██\033[0m"
            done
            echo
        done
    done
    echo
}

# Run all tests
echo "Running Terminal Tests..."
test_ansi_colors
test_block_elements
test_braille_patterns
test_cursor_positioning
test_sixel_graphics
test_truecolor
echo "Tests completed!"

