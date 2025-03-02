#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: term_color.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-27 06:14:43
#!/usr/bin/env python3
#
# term_color.py

from math import sqrt
from typing import Tuple, Dict, List, Union

class TermColor:
    """Library for managing terminal colors and escape codes."""
    @staticmethod
    def terminal_palette() -> List[Tuple[int, int, int]]:
        """Generate the 256-color terminal palette."""
        colors = []

        # 16 basic ANSI colors
        ansi_colors = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
            (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
            (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
        ]
        colors.extend(ansi_colors)

        # 6x6x6 RGB cube
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    colors.append((
                        r * 51,  # 51 = 255/5
                        g * 51,
                        b * 51,
                    ))

        # Grayscale (24 shades)
        for gray in range(24):
            level = 8 + gray * 10
            colors.append((level, level, level))

        return colors


    @staticmethod
    def rgb_to_hex(rgb_color) -> str:
        if isinstance(rgb_color, str):
            # Handle strings with various separators (e.g., "0,0,0" or "0 0 0")
            rgb_color = tuple(map(int, rgb_color.replace(",", " ").split()))
        elif isinstance(rgb_color, (list, set)):
            # Convert list or set to a tuple
            rgb_color = tuple(rgb_color)

        # Ensure it's a tuple with three elements
        if isinstance(rgb_color, tuple) and len(rgb_color) == 3:
            return "#{:02x}{:02x}{:02x}".format(*rgb_color)

        raise ValueError("Invalid RGB input. Must be a string, tuple, list, or set with three elements.")

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert a hex color to an RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        """Calculate the Euclidean distance between two colors."""
        return sqrt(sum((comp1 - comp2) ** 2 for comp1, comp2 in zip(c1, c2)))

    @classmethod
    def get(cls, mode: str| int) -> Dict[str, List[Dict[str, Union[str, Tuple[int, int, int]]]]]:
        """
        Get all escape codes for a given mode.

        Args:
            mode (str): Color mode ('8', '16', '256', 'truecolor').

        Returns:
            dict: Dictionary of foreground and background escape codes.
        """
        mode = str(mode)
        palette = []
        if mode in ["8", "16"]:
            palette = cls.terminal_palette()[:16]
        elif mode == "256":
            palette = cls.terminal_palette()
        elif mode == "truecolor":
            raise ValueError("get(mode) does not generate all 16M colors for 'truecolor'.")
        else:
            raise ValueError("Unsupported mode. Choose '8', '16', or '256'.")

        result = []
        for index, rgb in enumerate(palette):
            fg_escape = f"\033[38;5;{index}m"
            bg_escape = f"\033[48;5;{index}m"
            result.append({
                "index": index,
                "rgb": rgb,
                "foreground_escape": fg_escape,
                "background_escape": bg_escape,
            })

        return {"colors": result}

    @classmethod
    def display(cls, mode: str | int):
        """
        Render a visual reference for colors in the given mode.

        Args:
            mode (str): Color mode ('8', '16', '256').
        """
        mode = str(mode)
        palette = cls.get(mode)["colors"]
        print(f"Displaying colors for mode: {mode}\n")
        for color in palette:
            fg = color["foreground_escape"]
            bg = color["background_escape"]
            reset = "\033[0m"
            index = color["index"]
            rgb = color["rgb"]
            print(f"{fg}{bg} {index:3} RGB: {rgb} {reset}")

    @classmethod
    def closest_terminal_color(cls, hex_color: str, mode: str | int = 256) -> Dict[str, Union[str, int, Tuple[int, int, int]]]:
        """
        Find the closest terminal color for the given hex color.

        Args:
            hex_color (str): Hexadecimal color code.
            mode (str): Color mode ('16', '256', 'truecolor').

        Returns:
            dict: Dictionary with hex, RGB, terminal index, and escape sequences.
        """
        mode = str(mode)
        rgb = cls.hex_to_rgb(hex_color)
        if mode == "16":
            # Match against the 16 ANSI colors
            palette = cls.terminal_palette()[:16]
        elif mode == "256":
            # Match against the full 256-color palette
            palette = cls.terminal_palette()
        elif mode == "truecolor":
            # Use 24-bit truecolor directly
            fg_escape = f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
            bg_escape = f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
            reset_escape = "\033[0m"
            return {
                "input_hex": hex_color,
                "input_rgb": rgb,
                "foreground_escape": fg_escape,
                "background_escape": bg_escape,
                "reset_escape": reset_escape,
            }
        else:
            raise ValueError("Unsupported mode. Choose '16', '256', or 'truecolor'.")

        # Find the closest color in the chosen palette
        closest_color_index, closest_color_rgb = min(
            enumerate(palette),
            key=lambda c: cls.color_distance(rgb, c[1])
        )
        # Generate escape sequences
        fg_escape = f"\033[38;5;{closest_color_index}m"
        bg_escape = f"\033[48;5;{closest_color_index}m"
        reset_escape = "\033[0m"
        return {
            "input_hex": hex_color,
            "input_rgb": rgb,
            "closest_index": closest_color_index,
            "closest_rgb": closest_color_rgb,
            "closest_hex": "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2]),
            "foreground_escape": fg_escape,
            "background_escape": bg_escape,
            "reset_escape": reset_escape,
        }

def main():
    # Example hex color to test
    hex_color = "#4682b4"
    TC = TermColor()
    result = TC.closest_terminal_color(hex_color)

    TC.display(256)

    # Print the results
    print(f"Input Hex Color: {result['input_hex']}")
    print(f"Input RGB Color: {result['input_rgb']}")
    print(f"Closest Terminal Index: {result['closest_index']}")
    print(f"Closest Terminal HEX: {result['closest_hex']}")
    print(f"Closest Terminal RGB: {result['closest_rgb']}")
    print(f"Foreground Escape Sequence: {result['foreground_escape']}")
    print(f"Background Escape Sequence: {result['background_escape']}")
    print(f"Reset Escape Sequence: {result['reset_escape']}")

    # Example usage of escape sequences
    print(f"{result['foreground_escape']}This text is in the closest foreground color{result['reset_escape']}")
    print(f"{result['background_escape']}This text has the closest background color{result['reset_escape']}")

    DEFAULT_STYLES = {
        "email": "#ffd700",  # Gold - Highlights email addresses distinctly
        "url": "#00ff7f",  # SpringGreen - Stands out and indicates interactivity
        "unix_path": "#00ced1",  # DarkTurquoise - Represents system paths elegantly
        "windows_path": "74",  # SteelBlue - Differentiate from Unix paths
        "ini": "#ffa07a",  # LightSalmon - Soft highlight for INI sections
        "json": "#ff8c00",  # DarkOrange - JSON keys/values are critical, so bold
        "ipv4": "#1e90ff",  # DodgerBlue - Clear and sharp for IPv4 addresses
        "ipv6": "#9400d3",  # DarkViolet - Differentiates from IPv4
        "mac": "#32cd32",  # LimeGreen - Bright and clear for MAC addresses
        "timestamp": "#ff4500",  # OrangeRed - Stands out for timestamps
        "hex_number": "#daa520",  # GoldenRod - Distinguishable for hexadecimal
        "env_var": "#ff69b4",  # HotPink - Bold for environment variables
        "uuid": "#ff1493",  # DeepPink - Unique and eye-catching for UUIDs
    }
    for key, value in DEFAULT_STYLES.items():
        try:
            result = TC.closest_terminal_color(value)
            print(key, result['closest_index'])
            print(f"{result['foreground_escape']}This text is in the closest foreground color{result['reset_escape']}")
        except:
            pass

if __name__ == "__main__":
    main()
