#!/usr/bin/env python3
#
# /tmp/logo.py

import os
import shutil
import requests
from PIL import Image
from io import BytesIO
import pyfiglet

class symLogo():
    def __init__(self):
        # ASCII characters to use, ordered by brightness
        self.ASCII_CHARS = "@%#*+=-:. "
        print(self.image_to_ascii("https://smallroom.com/logo.png")) 

    def map_to_ascii(self, brightness):
        """Map a pixel's brightness to an ASCII character."""
        if brightness > 75:
            return "\033[93;40m" + self.ASCII_CHARS[brightness // 32] + "\033[0m"  # yellow for the symbol
        else:
            return "\033[30;40m" + self.ASCII_CHARS[brightness // 32] + "\033[0m"  # green for the background

    def image_to_ascii(self, url):
        """Convert an image to ASCII art."""
        response = requests.get(url, timeout=20)
        image = Image.open(BytesIO(response.content)).convert("L")  # convert image to grayscale

        # Resize the image to fit within the terminal width
        (width, height) = image.size
        aspect_ratio = height/width
        new_width = shutil.get_terminal_size().columns 
        new_height = aspect_ratio * new_width * 0.55
        image = image.resize((new_width, int(new_height)))

        # Convert each pixel to ASCII
        ascii_str = ""
        for y in range(image.height):
            for x in range(image.width):
                brightness = image.getpixel((x, y))
                ascii_str += self.map_to_ascii(brightness)
            ascii_str += "\n"

        return ascii_str

if __name__ == "__main__":
    symLogo()
