#!/usr/bin/env python3
#
# vision4.py

import pygame
import numpy as np
import matplotlib.pyplot as plt

# Define the size of the image and frames per second
width, height = 2048, 1024
fps = 60

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Define the color map for the topographic overlay
cmap = plt.get_cmap('terrain')

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate random pixel values for grayscale static and invert colors
    gray_data = 255 - np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # Normalize the grayscale data to the range [0, 1]
    normalized_data = gray_data / 255.0

    # Apply the color map to create a topographic overlay
    topo_data = cmap(normalized_data)

    # Convert the topographic data to an 8-bit RGB image
    topo_image = pygame.surfarray.make_surface(topo_data[:, :, :3] * 255)
    screen.blit(pygame.transform.scale(topo_image, (width, height)), (0, 0))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
