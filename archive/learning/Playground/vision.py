#!/usr/bin/env python3
#
# vision.py

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

    # Generate random pixel values for grayscale static and invert colors
    gray_data = 255 - np.random.randint(0, 256, (height, width), dtype=np.uint8)
    gray_image = pygame.surfarray.make_surface(gray_data)
    screen.blit(pygame.transform.scale(gray_image, (width, height)), (0, 0))

    # Generate random pixel values for RGB static and invert colors
    rgb_data = 255 - np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    rgb_image = pygame.surfarray.make_surface(rgb_data)
    screen.blit(pygame.transform.scale(rgb_image, (width, height)), (0, 0))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()

