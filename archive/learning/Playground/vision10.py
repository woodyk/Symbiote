#!/usr/bin/env python3
#
# vision10.py

import pygame
import numpy as np
import cv2

# Define the size of the image and frames per second
width, height = 1024, 768
fps = 30

import pygame
import numpy as np
import cv2

# Define the size of the image and frames per second
width, height = 1024, 768
fps = 30

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Initialize the threshold and contour thickness
threshold = 127
thickness = 3

# Define the size of the grid
grid_size = 10  # Reduced from 20 to 10 for a tighter grid

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                threshold = min(255, threshold + 10)
            elif event.key == pygame.K_DOWN:
                threshold = max(0, threshold - 10)
            elif event.key == pygame.K_LEFT:
                thickness = max(1, thickness - 1)
            elif event.key == pygame.K_RIGHT:
                thickness = min(10, thickness + 1)

    # Generate random pixel values for grayscale static and invert colors
    gray_data = 255 - np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # Apply a threshold to the grayscale data to create a binary image
    _, binary_data = cv2.threshold(gray_data, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the image with red color and adjusted thickness
    contour_image = cv2.drawContours(gray_data, contours, -1, (255, 0, 0), thickness)

    # Draw a grid on the image with green color
    for i in range(0, width, grid_size):
        cv2.line(contour_image, (i, 0), (i, height), (0, 255, 0), 1)  # Changed color to green
    for i in range(0, height, grid_size):
        cv2.line(contour_image, (0, i), (width, i), (0, 255, 0), 1)  # Changed color to green

    # Create a surface from the contour data
    contour_surface = pygame.surfarray.make_surface(contour_image)
    screen.blit(pygame.transform.scale(contour_surface, (width, height)), (0, 0))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
