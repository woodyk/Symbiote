#!/usr/bin/env python3
#
# vision3.py

import pygame
import numpy as np
from scipy.ndimage import convolve

# Define the size of the image and frames per second
width, height = 2048, 1024
fps = 10 

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Define the convolution filter
filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate random pixel values for grayscale static and invert colors
    gray_data = 255 - np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # Apply the convolution filter to create an outline effect
    outline_data = convolve(gray_data, filter)
    outline_data = np.clip(outline_data, 0, 255).astype(np.uint8)

    # Create a surface from the outline data
    outline_image = pygame.surfarray.make_surface(outline_data)
    screen.blit(pygame.transform.scale(outline_image, (width, height)), (0, 0))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
