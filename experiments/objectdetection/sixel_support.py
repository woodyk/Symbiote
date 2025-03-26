#!/usr/bin/env python3
#
# sixel_support.py

import numpy as np
from PIL import Image, ImageDraw
import sixel
import io

def generate_gradient_image(width, height):
    """Generates a color gradient image."""
    img = Image.new('RGB', (width, height))
    for y in range(height):
        for x in range(width):
            img.putpixel((x, y), (int(x * 255 / width), int(y * 255 / height), 128))
    return img

def generate_graph_image(width, height):
    """Generates a simple sine wave graph."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Draw the sine wave
    for x in range(width):
        y = int((height / 2) + (height / 4) * np.sin(2 * np.pi * x / width))
        draw.point((x, y), fill='blue')

    return img

def generate_simple_shapes_image(width, height):
    """Generates an image with simple shapes."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Draw a red circle
    draw.ellipse((width//4 - 50, height//4 - 50, width//4 + 50, height//4 + 50), outline='red', width=5)

    # Draw a green rectangle
    draw.rectangle((3*width//4 - 50, 3*height//4 - 50, 3*width//4 + 50, 3*height//4 + 50), outline='green', width=5)

    return img

def render_image_in_sixel(image):
    """Converts a PIL image to Sixel and prints it in the terminal."""
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)

    # Using SixelWriter to render the image in the terminal
    writer = sixel.SixelWriter()
    writer.draw(output)

def test_sixel_support():
    """Run multiple tests to check terminal's Sixel support."""
    print("Testing Sixel Support...")

    width, height = 320, 240  # Resolution for images

    # Test 1: Color Gradient
    print("Rendering color gradient...")
    gradient_img = generate_gradient_image(width, height)
    render_image_in_sixel(gradient_img)

    # Test 2: Sine Wave Graph
    print("Rendering sine wave graph...")
    graph_img = generate_graph_image(width, height)
    render_image_in_sixel(graph_img)

    # Test 3: Simple Shapes
    print("Rendering simple shapes...")
    shapes_img = generate_simple_shapes_image(width, height)
    render_image_in_sixel(shapes_img)

if __name__ == "__main__":
    test_sixel_support()

