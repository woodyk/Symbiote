#!/usr/bin/env python3
#
# vision29.py

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

# Initialize the channel number and noise level
channel = 0
noise_level = 10 

# Initialize the font
font = pygame.font.SysFont('courier', 30)

# Define the controls
controls = ['UP: Increase noise', 'DOWN: Decrease noise', 'LEFT: Previous channel', 'RIGHT: Next channel', 'C: Change contour color', '+: Increase bubble size', '-: Decrease bubble size']

# Initialize the help menu visibility
show_help = False

# Initialize the contour color
contour_color = (225, 255, 255)

# Initialize the bubble size
bubble_size = .25 

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                noise_level = min(256, noise_level + 10)
            elif event.key == pygame.K_DOWN:
                noise_level = max(1, noise_level - 10)  # Set the lower limit to 1
            elif event.key == pygame.K_LEFT:
                channel = max(0, channel - 1)
            elif event.key == pygame.K_RIGHT:
                channel = min(56, channel + 1)
            elif event.key == pygame.K_h:
                show_help = not show_help
            elif event.key == pygame.K_c:
                contour_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                bubble_size += 1
            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                bubble_size = max(1, bubble_size - 1)

    # Generate random pixel values for grayscale static
    gray_data = np.random.randint(0, noise_level, (height, width), dtype=np.uint8)

    # Find contours in the grayscale data
    contours, _ = cv2.findContours(gray_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dilate the contours to increase their size
    dilated = cv2.dilate(gray_data, None, iterations=bubble_size)

    # Draw the contours on the grayscale data
    cv2.drawContours(dilated, contours, -1, contour_color, 3)

    # Create a surface from the grayscale data
    gray_surface = pygame.surfarray.make_surface(dilated)
    screen.blit(pygame.transform.scale(gray_surface, (width, height)), (0, 0))

    # Display the channel number
    channel_text = font.render(str(channel), True, (0, 255, 0))
    screen.blit(channel_text, (width - channel_text.get_width() - 10, height - channel_text.get_height() - 10))

    # Display the controls if the help menu is shown
    if show_help:
        for i, control in enumerate(controls):
            control_text = font.render(control, True, (255, 255, 255))
            screen.blit(control_text, (10, height - (len(controls) - i) * control_text.get_height()))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
