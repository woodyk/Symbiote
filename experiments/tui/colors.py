#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: colors.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-30 04:00:22
# Modified: 2024-11-30 13:26:46

import os
import math
import time
import random
import shutil

def generate_braille_gradient(x_width, y_height):
    """
    Renders a beautiful gradient using braille characters and 256-bit color ASCII escape codes.

    Args:
        x_width (int): Width of the block in braille characters.
        y_height (int): Height of the block in braille characters.
    """
    def rgb_to_ansi256(r, g, b):
        """Convert RGB to the nearest ANSI 256 color code."""
        if r == g == b:
            if r < 8:
                return 16
            if r > 248:
                return 231
            return int(((r - 8) / 247) * 24) + 232
        return 16 + (36 * (r // 51)) + (6 * (g // 51)) + (b // 51)

    def color_gradient(x, y, max_x, max_y):
        """Generate RGB values based on position for a gradient effect."""
        r = int(255 * (x / max_x))
        g = int(255 * (y / max_y))
        b = int(255 * ((max_x - x) / max_x))
        return r, g, b

    braille_char = "⣿"  # Full block Braille character.

    gradient = ""
    for y in range(y_height):
        for x in range(x_width):
            r, g, b = color_gradient(x, y, x_width, y_height)
            ansi_code = rgb_to_ansi256(r, g, b)
            gradient += f"\033[38;5;{ansi_code}m{braille_char}\033[0m"
        gradient += "\n"

    print(gradient)



# Gradient generation utility
def rgb_to_ansi256(r, g, b):
    """Convert RGB to the nearest ANSI 256 color code."""
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return int(((r - 8) / 247) * 24) + 232
    return 16 + (36 * (r // 51)) + (6 * (g // 51)) + (b // 51)

def generate_pattern(x_width, y_height, pattern_function, duration=1):
    """Generate a gradient pattern based on the provided function."""
    os.system('clear')
    braille_char = "⣿"  # Full block Braille character.
    output = ""
    for y in range(y_height):
        for x in range(x_width):
            r, g, b = pattern_function(x, y, x_width, y_height)
            ansi_code = rgb_to_ansi256(r, g, b)
            output += f"\033[38;5;{ansi_code}m{braille_char}\033[0m"
        output += "\n"
    print(output)
    time.sleep(duration)

def animated_pattern(x_width, y_height, pattern_function, duration=5):
    """Generate an animated gradient pattern."""
    frame = 0
    end_time = time.time() + duration
    while time.time() < end_time:
        os.system("clear")
        braille_char = "⣿"
        output = ""
        for y in range(y_height):
            for x in range(x_width):
                r, g, b = pattern_function(x, y, x_width, y_height, frame)
                ansi_code = rgb_to_ansi256(r, g, b)
                output += f"\033[38;5;{ansi_code}m{braille_char}\033[0m"
            output += "\n"
        print(output)
        frame += 1
        time.sleep(0.1)

# Patterns
def circular_gradient(x, y, max_x, max_y):
    center_x, center_y = max_x // 2, max_y // 2
    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = math.sqrt(center_x ** 2 + center_y ** 2)
    intensity = distance / max_distance
    r = int(255 * (1 - intensity))
    g = int(255 * intensity)
    b = int(255 * (1 - abs(0.5 - intensity) * 2))
    return r, g, b

def horizontal_stripes(x, y, max_x, max_y):
    stripe_height = max_y // 10
    if (y // stripe_height) % 2 == 0:
        return 255, 0, 0  # Red
    else:
        return 0, 0, 255  # Blue

def vertical_stripes(x, y, max_x, max_y):
    stripe_width = max_x // 10
    if (x // stripe_width) % 2 == 0:
        return 0, 255, 0  # Green
    else:
        return 255, 255, 0  # Yellow

def checkerboard(x, y, max_x, max_y):
    square_size = 4
    if (x // square_size + y // square_size) % 2 == 0:
        return 255, 0, 0  # Red
    else:
        return 0, 255, 0  # Green

def spiral_pattern(x, y, max_x, max_y, frame):
    center_x, center_y = max_x // 2, max_y // 2
    dx, dy = x - center_x, y - center_y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    angle = math.atan2(dy, dx)
    spiral_intensity = (distance + angle * 10 + frame) % 255
    r = int(spiral_intensity)
    g = int(255 - spiral_intensity)
    b = int((spiral_intensity * 0.5) % 255)
    return r, g, b

def wave_pattern(x, y, max_x, max_y, frame):
    frequency = 10
    wave_intensity = (math.sin((x + frame) / frequency) + math.cos((y + frame) / frequency)) / 2
    r = int(255 * abs(wave_intensity))
    g = int(255 * (1 - abs(wave_intensity)))
    b = 128
    return r, g, b

def diagonal_gradient(x, y, max_x, max_y):
    intensity = (x + y) / (max_x + max_y)
    r = int(255 * intensity)
    g = int(255 * (1 - intensity))
    b = int(128 * abs(0.5 - intensity) * 2)
    return r, g, b

def random_noise(x, y, max_x, max_y):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b

def sunset_gradient(x, y, max_x, max_y):
    intensity = y / max_y
    r = int(255)
    g = int(200 * intensity)
    b = int(100 * intensity)
    return r, g, b

def fire_effect(x, y, max_x, max_y, frame):
    intensity = random.random() * (1 - y / max_y)
    r = int(255 * intensity)
    g = int(150 * intensity)
    b = int(50 * intensity)
    return r, g, b

def matrix_rain(x, y, max_x, max_y, frame):
    column_seed = (x + frame) % max_y
    fade = (y - column_seed) % max_y
    intensity = max(0, 1 - (fade / 10))
    g = int(255 * intensity)
    return 0, g, 0

def scrolling_rainbow(x, y, max_x, max_y, frame):
    frequency = 0.1
    r = int(127 * (math.sin(frequency * (x + frame)) + 1))
    g = int(127 * (math.sin(frequency * (x + frame) + 2) + 1))
    b = int(127 * (math.sin(frequency * (x + frame) + 4) + 1))
    return r, g, b

def water_ripple(x, y, max_x, max_y, frame):
    center_x, center_y = max_x // 2, max_y // 2
    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    ripple = math.sin(distance / 2 - frame / 5) * 0.5 + 0.5
    r = int(0)
    g = int(100 * ripple)
    b = int(255 * ripple)
    return r, g, b

# Main demonstration function
def main():
    term_size = shutil.get_terminal_size()
    x_width = term_size.columns * 2  # Each braille character is twice as wide
    y_height = term_size.lines - 2  # Leave some margin for aesthetics
    pause_duration = 2  # Seconds to pause between patterns

    patterns = [
        ("Circular Gradient", circular_gradient, False),
        ("Horizontal Stripes", horizontal_stripes, False),
        ("Vertical Stripes", vertical_stripes, False),
        ("Checkerboard", checkerboard, False),
        ("Spiral Pattern", spiral_pattern, True),
        ("Wave Pattern", wave_pattern, True),
        ("Diagonal Gradient", diagonal_gradient, False),
        ("Random Noise", random_noise, False),
        ("Sunset Gradient", sunset_gradient, False),
        ("Fire Effect", fire_effect, True),
        ("Matrix Rain", matrix_rain, True),
        ("Scrolling Rainbow", scrolling_rainbow, True),
        ("Water Ripple", water_ripple, True),
    ]
    generate_braille_gradient(40, 20)

    for name, pattern, is_animated in patterns:
        print(f"Rendering: {name}")
        time.sleep(1)
        if is_animated:
            animated_pattern(x_width, y_height, pattern, duration=5)
        else:
            generate_pattern(x_width, y_height, pattern, duration=pause_duration)

if __name__ == "__main__":
   main()

import sys
import os
import time

def render_symbiote_banner():
    """Render the Symbiote banner with color effects."""
    banner_lines = [
        "\033[38;5;82m  ___ _   _ _ __ ___ | |__ (_) ___ | |_ ___  \033[0m",
        "\033[38;5;118m / __| | | | '_ ` _ \\| '_ \\| |/ _ \\| __/ _ \\ \033[0m",
        "\033[38;5;154m \\__ \\ |_| | | | | | | |_) | | (_) | ||  __/ \033[0m",
        "\033[38;5;190m |___/\\__, |_| |_| |_|_.__/|_|\\___/ \\__\\___| \033[0m",
        "\033[38;5;226m      |___/                                 \033[0m",
        "",
        "\033[38;5;46m     ███████╗██╗   ██╗███╗   ███╗██████╗ ██╗ ██████╗ ████████╗███████╗\033[0m",
        "\033[38;5;82m     ██╔════╝██║   ██║████╗ ████║██╔══██╗██║██╔═══██╗╚══██╔══╝██╔════╝\033[0m",
        "\033[38;5;118m     █████╗  ██║   ██║██╔████╔██║██████╔╝██║██║   ██║   ██║   █████╗  \033[0m",
        "\033[38;5;154m     ██╔══╝  ██║   ██║██║╚██╔╝██║██╔═══╝ ██║██║   ██║   ██║   ██╔══╝  \033[0m",
        "\033[38;5;190m     ███████╗╚██████╔╝██║ ╚═╝ ██║██║     ██║╚██████╔╝   ██║   ███████╗\033[0m",
        "\033[38;5;226m     ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚══════╝\033[0m",
        "",
        "\033[38;5;14m                  ██████╗ ███╗   ██╗███╗   ███╗██╗██████╗ ███████╗\033[0m",
        "\033[38;5;27m                 ██╔═══██╗████╗  ██║████╗ ████║██║██╔══██╗██╔════╝\033[0m",
        "\033[38;5;39m                 ██║   ██║██╔██╗ ██║██╔████╔██║██║██████╔╝█████╗  \033[0m",
        "\033[38;5;51m                 ██║   ██║██║╚██╗██║██║╚██╔╝██║██║██╔═══╝ ██╔══╝  \033[0m",
        "\033[38;5;45m                 ╚██████╔╝██║ ╚████║██║ ╚═╝ ██║██║██║     ███████╗\033[0m",
        "\033[38;5;33m                  ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚═╝╚═╝     ╚══════╝\033[0m",
    ]

    # Write each line to stdout
    for line in banner_lines:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()
render_symbiote_banner()
time.sleep(3)  # Pause to admire the banner

