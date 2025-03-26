#!/usr/bin/env python3
#
# qr_generator.py

import qrcode
from PIL import Image, ImageDraw, ImageColor

def show_gradient_qr_code(
    text: str,
    center_color: str = "#00FF00",  # Lime color in hex
    outer_color: str = "#0000FF",   # Blue color in hex
    back_color: str = "black",
    dot_size: int = 10,
    border_size: int = 10
):
    # Create a QR code object
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # Higher error correction
        box_size=dot_size,
        border=4,
    )

    # Add data to the QR code
    qr.add_data(text)
    qr.make(fit=True)

    # Get the QR code matrix
    qr_matrix = qr.get_matrix()
    qr_size = len(qr_matrix)

    # Create a new image with a background color
    img_size = qr_size * dot_size + 2 * border_size
    img = Image.new("RGBA", (img_size, img_size), back_color)
    draw = ImageDraw.Draw(img)

    # Calculate the gradient
    def get_gradient_color(x, y, center_x, center_y, inner_color, outer_color, max_dist):
        # Distance from the center
        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        ratio = min(dist / max_dist, 1)

        # Interpolating the color
        r1, g1, b1 = inner_color
        r2, g2, b2 = outer_color

        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)

        return (r, g, b)

    # Center of the QR code
    center_x = img_size // 2
    center_y = img_size // 2
    max_dist = ((center_x) ** 2 + (center_y) ** 2) ** 0.5

    # Convert hex colors to RGB
    center_color_rgb = ImageColor.getrgb(center_color)
    outer_color_rgb = ImageColor.getrgb(outer_color)

    # Draw the QR code with gradient dots
    for y in range(qr_size):
        for x in range(qr_size):
            if qr_matrix[y][x]:
                x1 = x * dot_size + border_size
                y1 = y * dot_size + border_size
                fill_color = get_gradient_color(x1, y1, center_x, center_y, center_color_rgb, outer_color_rgb, max_dist)
                draw.ellipse([x1, y1, x1 + dot_size, y1 + dot_size], fill=fill_color)

    # Display the image
    img.show()

# Example usage with a gradient effect
show_gradient_qr_code(
    text="https://example.com",
    center_color="#00FF00",  # Lime
    outer_color="#0000FF",   # Blue
    back_color="black"
)

show_gradient_qr_code(text="hello world")
