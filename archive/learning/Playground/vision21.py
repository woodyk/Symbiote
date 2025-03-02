#!/usr/bin/env python3
#
# vision21.py

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

    # Create a separate image for the contours
    contour_image = np.zeros_like(gray_data)

    # Draw the contours on the contour image with black color and increased thickness
    cv2.drawContours(contour_image, contours, -1, 0, 3)

    # Create a mask from the binary image
    mask = np.ones_like(gray_data) * 255
    cv2.drawContours(mask, contours, -1, 0, -1)

    # Convert the mask to a 3-channel image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Convert the grayscale data to a 3-channel image
    gray_data = cv2.cvtColor(gray_data, cv2.COLOR_GRAY2BGR)

    # Blend the mask with the original image
    blended_image = cv2.addWeighted(gray_data, 0.7, mask, 0.3, 0)

    # Create a surface from the blended image
    blended_surface = pygame.surfarray.make_surface(blended_image)
    screen.blit(pygame.transform.scale(blended_surface, (width, height)), (0, 0))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
