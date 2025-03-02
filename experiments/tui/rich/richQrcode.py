#!/usr/bin/env python3
#
# richQrcode.py

import qrcode
from rich.console import Console
from rich.text import Text


def generate_qr_terminal(
    text: str,
    center_color: str = "#00FF00",
    outer_color: str = "#0000FF",
    back_color: str = "#000000",
):
    console = Console()

    # Generate QR code matrix
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=1,
        border=4,  # Ensure QR standard border
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_matrix = qr.get_matrix()

    # Dimensions of the QR code
    qr_size = len(qr_matrix)

    # Convert hex colors to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    center_rgb = hex_to_rgb(center_color)
    outer_rgb = hex_to_rgb(outer_color)

    # Interpolate between colors
    def interpolate_color(x, y, center_x, center_y, max_dist):
        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        ratio = min(dist / max_dist, 1)

        r = int(center_rgb[0] + ratio * (outer_rgb[0] - center_rgb[0]))
        g = int(center_rgb[1] + ratio * (outer_rgb[1] - center_rgb[1]))
        b = int(center_rgb[2] + ratio * (outer_rgb[2] - center_rgb[2]))

        return f"rgb({r},{g},{b})"

    # Calculate center and max distance
    center_x = qr_size // 2
    center_y = qr_size // 2
    max_dist = ((center_x) ** 2 + (center_y) ** 2) ** 0.5

    # Prepare to center the QR code in the terminal
    terminal_width = console.size.width
    qr_width = qr_size * 2  # Each QR column rendered takes 2 spaces in terminal
    padding = max((terminal_width - qr_width) // 2, 0)

    # Render QR code as a square
    qr_render = []
    for y in range(0, qr_size, 2):  # Process two rows at a time
        line = Text(" " * padding)  # Add left padding to center the QR code
        for x in range(qr_size):
            upper = qr_matrix[y][x]
            lower = qr_matrix[y + 1][x] if y + 1 < qr_size else 0

            if upper and lower:
                color = interpolate_color(x, y, center_x, center_y, max_dist)
                line.append("█", style=f"{color} on {color}")
            elif upper:
                color = interpolate_color(x, y, center_x, center_y, max_dist)
                line.append("▀", style=f"{color} on {back_color}")
            elif lower:
                color = interpolate_color(x, y + 1, center_x, center_y, max_dist)
                line.append("▄", style=f"{color} on {back_color}")
            else:
                line.append(" ", style=f"{back_color} on {back_color}")
        console.print(line)
        qr_render.append(line)


# Example Usage
generate_qr_terminal(
    "https://example.com",
    center_color="#00FF00",
    outer_color="#0000FF",
    back_color="#000000",
)

