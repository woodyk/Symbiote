#!/usr/bin/env python3
#
# vision200.py

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

# Maximum number of frames a contour will persist for
contour_persistence = 10

# List of surfaces that hold previous frames' contour lines
contour_surfaces = []

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate random pixel values for grayscale static and invert colors
    gray_data = 255 - np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # Apply a threshold to the grayscale data to create a binary image
    _, binary_data = cv2.threshold(gray_data, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty black image to draw the contours on
    contour_image = np.zeros((height, width), dtype=np.uint8)

    # Draw the contours on the image with white color and increased thickness
    cv2.drawContours(contour_image, contours, -1, (255), 3)

    # Create a surface from the contour data
    contour_surface = pygame.surfarray.make_surface(contour_image)

    # Add the new surface to the list
    contour_surfaces.append(contour_surface)

    # If there are too many surfaces in the list, remove the oldest
    if len(contour_surfaces) > contour_persistence:
        contour_surfaces.pop(0)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw each surface in the list on the screen, with decreasing opacity
    for i, contour_surface in enumerate(contour_surfaces):
        contour_surface.set_alpha(255 * (i+1) // len(contour_surfaces))
        screen.blit(pygame.transform.scale(contour_surface, (width, height)), (0, 0))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
