#!/usr/bin/env python3
#
# rgb_vision.py

import pygame
import numpy as np

# Define the size of the image and frames per second
width, height = 2048, 1024
fps = 60

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate random pixel values for grayscale static
    gray_data = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # Create a surface from the grayscale data
    gray_image = pygame.surfarray.make_surface(np.stack([gray_data]*3, axis=-1))

    # Create a "density" effect by drawing larger circles for brighter pixels
    for y in range(height - 1):
        for x in range(width - 1):
            color = gray_image.get_at((x, y))
            brightness = (color.r + color.g + color.b) / 3
            if brightness > 128:
                pygame.draw.circle(screen, color, (x, y), int(brightness / 256 * 10))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
