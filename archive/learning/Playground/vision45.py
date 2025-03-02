#!/usr/bin/env python3
#
# vision45.py

import pygame
import numpy as np
import cv2

# Initialize Pygame
pygame.init()

# Define the size of the image and frames per second
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# Initialize the noise level, decay factor, and persistence
noise_level = 50
decay_factor = 0.9
persistence = 500 

# Create a black image to start with
image = np.zeros((height, width), dtype=np.uint8)

# Create a history image to store the sum of the pixel values over the past several frames
history = np.zeros((height, width), dtype=np.uint8)

# Create a structuring element for the morphological operations
kernel = np.ones((5,5),np.uint8)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate random pixel values for grayscale static
    noise = np.random.randint(0, noise_level, (height, width), dtype=np.uint8)

    # Add the noise to the image
    image = cv2.add(image, noise)

    # Apply a threshold to the image
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological opening to reduce noise
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Add the image to the history
    history = cv2.add(history, image)

    # Decay the history
    history = cv2.multiply(history, decay_factor, dtype=cv2.CV_32F).astype(np.uint8)

    # Apply a colormap to the history
    history_color = cv2.applyColorMap(history, cv2.COLORMAP_JET)

    # Create a Pygame surface from the history
    surface = pygame.surfarray.make_surface(history_color)

    # Blit the surface onto the screen
    screen.blit(pygame.transform.scale(surface, (width, height)), (0, 0))

    pygame.display.flip()

pygame.quit()
