#!/usr/bin/env python3
#
# vision5.py

import pygame
import numpy as np
import cv2

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

    # Apply the Canny edge detection algorithm
    edges = cv2.Canny(gray_data, 100, 200)

    # Create a surface from the edge data
    edge_image = pygame.surfarray.make_surface(edges)
    screen.blit(pygame.transform.scale(edge_image, (width, height)), (0, 0))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
