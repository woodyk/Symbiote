#!/usr/bin/env python3
#
# vision44.py

import pygame
import numpy as np
import cv2

# Define the size of the image and frames per second
width, height = 2056, 1024
fps = 60

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Initialize the threshold, contour thickness, noise level, dilation/erosion size, contour persistence, frame persistence, and decay factor
threshold = 127
thickness = 1
noise_level = 256
morph_size = 0
contour_persistence = 120 
frame_persistence = 120
decay_factor = 0.1  # Adjust this value to change the rate of decay

# Initialize the font
font = pygame.font.SysFont('courier', 30)

# Define the controls
controls = ['UP: Increase threshold', 'DOWN: Decrease threshold', 'LEFT: Decrease thickness', 'RIGHT: Increase thickness', 'W: Increase noise', 'S: Decrease noise', 'A: Increase negative space', 'D: Decrease negative space', 'H: Show/Hide this help', 'M: Toggle heat map', 'SPACE: Pause/Resume', 'Q: Increase contour persistence', 'E: Decrease contour persistence', 'R: Increase frame persistence', 'F: Decrease frame persistence']

# Initialize the help menu visibility, heat map, pause, contour list, and overlay
show_help = False
show_heat_map = False
paused = False
contours_list = []
overlay = np.zeros((height, width), dtype=np.uint8)  # Initialize the overlay with black pixels

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
            elif event.key == pygame.K_w:
                noise_level = min(256, noise_level + 10)
            elif event.key == pygame.K_s:
                noise_level = max(0, noise_level - 10)
            elif event.key == pygame.K_a:
                morph_size = min(10, morph_size + 1)
            elif event.key == pygame.K_d:
                morph_size = max(0, morph_size - 1)
            elif event.key == pygame.K_h:
                show_help = not show_help
            elif event.key == pygame.K_m:
                show_heat_map = not show_heat_map
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_q:
                contour_persistence = min(100, contour_persistence + 1)
            elif event.key == pygame.K_e:
                contour_persistence = max(1, contour_persistence - 1)
            elif event.key == pygame.K_r:
                frame_persistence = min(240, frame_persistence + 10)
            elif event.key == pygame.K_f:
                frame_persistence = max(1, frame_persistence - 10)

    if not paused:
        # Generate random pixel values for grayscale static and invert colors
        gray_data = 255 - np.random.randint(0, noise_level, (height, width), dtype=np.uint8)

        # Apply a Sobel operator to the image to highlight edges
        #sobelx = cv2.Sobel(gray_data, cv2.CV_64F, 1, 0, ksize=5)
        #sobely = cv2.Sobel(gray_data, cv2.CV_64F, 0, 1, ksize=5)
        #gray_data = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2.0), 0.5, cv2.pow(sobely, 2.0), 0.5, 0)).astype(np.uint8)

        # Apply a threshold to the grayscale data to create a binary image
        #_, binary_data = cv2.threshold(gray_data, threshold, 255, cv2.THRESH_BINARY)

        # Erode the binary image to increase the negative space
        #eroded_data = cv2.erode(binary_data, None, iterations=morph_size)

        # Find contours in the eroded image
        contours, _ = cv2.findContours(gray_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Add the new contours to the list
        contours_list.append(contours)

        # Remove contours that are too old
        while len(contours_list) > contour_persistence:
            contours_list.pop(0)

        # Draw all contours on the image with red color and adjusted thickness
        for contours in contours_list:
            gray_data = cv2.drawContours(gray_data, contours, -1, (255, 0, 0), thickness)

        # Update the overlay with the minimum pixel values
        overlay = np.minimum(overlay, gray_data)

        # Decay the overlay
        overlay = (overlay * decay_factor).astype(np.uint8)

        # Apply a heat map if enabled
        if show_heat_map:
            overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)

        # Create a surface from the overlay
        #overlay_surface = pygame.surfarray.make_surface(overlay)
        #screen.blit(pygame.transform.scale(overlay_surface, (width, height)), (0, 0))

    # Draw a solid white dot in the center of the screen
    pygame.draw.circle(screen, (0, 0, 0), (width // 2, height // 2), 3)

    # Display the controls if the help menu is shown
    if show_help:
        for i, control in enumerate(controls):
            control_text = font.render(control, True, (255, 255, 255))
            screen.blit(control_text, (10, height - (len(controls) - i) * control_text.get_height()))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
