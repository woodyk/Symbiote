#!/usr/bin/env python3
#
# color_finder.py

import re
from typing import Any, Dict, List, Tuple, Union


class ColorIdentifier:
    """
    A universal color identifier that can recognize and normalize colors in various formats.
    """

    @staticmethod
    def _is_hex_color(value: str) -> bool:
        """Check if the input is a valid hex color."""
        return isinstance(value, str) and re.match(r"^#?[0-9a-fA-F]{6}$", value.strip())

    @staticmethod
    def _is_rgb_string(value: str) -> bool:
        """Check if the input is an RGB string."""
        return isinstance(value, str) and re.match(r"^\s*[0-9a-fA-F]{1,2}[\s,]+[0-9a-fA-F]{1,2}[\s,]+[0-9a-fA-F]{1,2}\s*$", value)

    @staticmethod
    def _is_rgb_list(value: Union[List[int], Tuple[int], set]) -> bool:
        """Check if the input is a list, tuple, or set with RGB values."""
        return (
            isinstance(value, (list, set, tuple))
            and len(value) == 3
            and all(isinstance(v, int) and 0 <= v <= 255 for v in value)
        )

    @staticmethod
    def _is_integer_color(value: int) -> bool:
        """Check if the input is an integer that might represent an RGB triplet."""
        return isinstance(value, int) and 0 <= value <= 255255255

    @staticmethod
    def _extract_potential_colors(text: str) -> List[str]:
        """Extract potential color patterns from a string using regex."""
        regex_patterns = [
            r"#?[0-9a-fA-F]{6}",  # Hexadecimal colors
            r"\b\d{1,3}[\s,]+\d{1,3}[\s,]+\d{1,3}\b",  # RGB strings
        ]
        matches = []
        for pattern in regex_patterns:
            matches.extend(re.findall(pattern, text))
        return matches

    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert an RGB tuple to a hex color string."""
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert a hex color to an RGB tuple."""
        hex_color = hex_color.lstrip("#").strip()
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def identify_color(color_input: Any) -> Union[Dict[str, Any], None]:
        """
        Identify and normalize a single color input.

        Args:
            color_input: The color input in any format.

        Returns:
            dict: A dictionary with the type, RGB tuple, and hex value of the color.
        """
        if isinstance(color_input, str):
            color_input = color_input.strip()

            if ColorIdentifier._is_hex_color(color_input):
                rgb = ColorIdentifier._hex_to_rgb(color_input)
                return {"type": "hex", "rgb": rgb, "hex": ColorIdentifier._rgb_to_hex(rgb)}

            if ColorIdentifier._is_rgb_string(color_input):
                # Detect hexadecimal components in RGB string
                components = re.split(r"[\s,]+", color_input)
                if all(re.match(r"^[0-9a-fA-F]{1,2}$", c) for c in components):
                    rgb = tuple(int(c, 16) for c in components)
                else:
                    rgb = tuple(int(c) for c in components)
                return {"type": "rgb_string", "rgb": rgb, "hex": ColorIdentifier._rgb_to_hex(rgb)}

        if isinstance(color_input, (list, tuple, set)) and len(color_input) == 3:
            if ColorIdentifier._is_rgb_list(color_input):
                rgb = tuple(color_input)
                return {"type": "rgb_list", "rgb": rgb, "hex": ColorIdentifier._rgb_to_hex(rgb)}

        if isinstance(color_input, int) and ColorIdentifier._is_integer_color(color_input):
            rgb = (
                (color_input // 1000000) % 256,
                (color_input // 1000) % 256,
                color_input % 256,
            )
            return {"type": "integer", "rgb": rgb, "hex": ColorIdentifier._rgb_to_hex(rgb)}

        return None

    @staticmethod
    def identify_colors_in_text(text: str) -> List[Dict[str, Any]]:
        """
        Identify all colors found in a large string of text.

        Args:
            text: The input string containing potential color representations.

        Returns:
            list: A list of dictionaries with color information for each match.
        """
        potential_colors = ColorIdentifier._extract_potential_colors(text)
        found_colors = []
        for color in potential_colors:
            identified = ColorIdentifier.identify_color(color)
            if identified:
                found_colors.append(identified)
        return found_colors

    @staticmethod
    def validate_color_input(color_input: Any) -> bool:
        """
        Check if the input can be identified as a valid color.

        Args:
            color_input: The input to validate.

        Returns:
            bool: True if the input is a valid color representation, False otherwise.
        """
        return ColorIdentifier.identify_color(color_input) is not None

print(ColorIdentifier.identify_color("46 2b fe"))
print(ColorIdentifier.identify_color("#4682b4"))
print(ColorIdentifier.identify_color(255000155))

