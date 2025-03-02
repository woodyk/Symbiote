#!/usr/bin/env python3
#
# vision30.py

import pygame
import numpy as np
import cv2

# Define the size of the image and frames per second
width, height = 800, 600 
fps = 20 

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Initialize the channel number and noise level
channel = 0
noise_level = 256 

# Initialize the font
font = pygame.font.SysFont('courier', 20)

# Define the controls
controls = ['UP: Increase noise', 'DOWN: Decrease noise', 'LEFT: Previous channel', 'RIGHT: Next channel']

# Initialize the background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Initialize a list to store the history of contours
contour_history = []

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

    # Generate random pixel values for grayscale static
    gray_data = np.random.randint(0, noise_level, (height, width), dtype=np.uint8)

    # Apply the background subtractor
    fgMask = backSub.apply(gray_data)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours based on their size
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Check if the contours are similar to the ones in the previous frame
    for cnt in contours:
        for old_cnt in contour_history:
            if cv2.matchShapes(cnt, old_cnt, cv2.CONTOURS_MATCH_I1, 0) < 0.1:
                # The contour is similar to one in the previous frame, so draw a convex hull around it
                hull = cv2.convexHull(cnt)
                cv2.drawContours(gray_data, [hull], 0, (0, 255, 0), 2)

    # Update the contour history
    contour_history = contours

    # Create a surface from the grayscale data
    gray_surface = pygame.surfarray.make_surface(gray_data)
    screen.blit(pygame.transform.scale(gray_surface, (width, height)), (0, 0))

    # Display the channel number
    channel_text = font.render(str(channel), True, (0, 255, 0))
    screen.blit(channel_text, (width - channel_text.get_width() - 10, height - channel_text.get_height() - 10))

    # Display the controls
    for i, control in enumerate(controls):
        control_text = font.render(control, True, (255, 255, 255))
        screen.blit(control_text, (10, height - (len(controls) - i) * control_text.get_height()))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
