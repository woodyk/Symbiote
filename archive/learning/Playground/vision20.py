#!/usr/bin/env python3
#
# vision20.py

import pygame
import numpy as np
import cv2

# Define the size of the image and frames per second
width, height = 2048, 768
fps = 30

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Initialize the threshold, contour thickness, and sensitivity
threshold = 127
thickness = 3
sensitivity = 10.0

# Define the size of the grid
grid_size = 10  # Reduced from 20 to 10 for a tighter grid

# Define the color scale for the heat map
color_scale = np.array([
    [0, 0, 0],  # Black
    [255, 255, 255]  # White
])

# Generate a 1024-color gradient
gradient = np.zeros((64, 1, 3), dtype=np.uint8)
for i in range(64):
    #t = i / 1023.0
    t = i / 64.0
    gradient[i, 0, :] = (1.0 - t) * color_scale[0] + t * color_scale[1]

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

    # Generate random pixel values for the color map and invert colors
    color_data = 255 - np.random.randint(0, 512, (height, width), dtype=np.uint16)

    # Normalize the color data to the range 0-1023
    color_data = (color_data * 1023 / 65535).astype(np.uint16)

    # Map the color data to the gradient and remove the extra dimension
    color_map = gradient[color_data].squeeze()

    # Apply a threshold to the color data to create a binary image
    _, binary_data = cv2.threshold(color_data, threshold, 1023, cv2.THRESH_BINARY)

    # Convert the binary image to an 8-bit single-channel image
    binary_data = (binary_data / 4).astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the image with red color and adjusted thickness
    contour_image = cv2.drawContours(color_data, contours, -1, (1023, 0, 0), thickness)

    # Calculate the depth map
    depth_map = np.zeros((height, width))
    for contour in contours:
        cv2.drawContours(depth_map, [contour], -1, cv2.contourArea(contour), -1)

    # Calculate the average color intensity of the contours
    avg_intensity = np.mean([cv2.mean(color_map, mask=cv2.drawContours(np.zeros_like(binary_data), [contour], -1, 255, -1)) for contour in contours])

    # Adjust the sensitivity based on the average color intensity
    sensitivity = avg_intensity / 255.0

    # Apply the sensitivity to the depth map
    depth_map *= sensitivity

    # Normalize the depth map to the range 0-1
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    # Map the depth map to the color scale
    depth_map_color = (depth_map * 255).astype(np.uint8)

    # Convert the depth map color to a 3-channel image
    depth_map_color = cv2.cvtColor(depth_map_color, cv2.COLOR_GRAY2BGR)

    # Blend the depth map with the original image
    blended_image = cv2.addWeighted(color_map, 0.8, depth_map_color, 0.2, 0)

    # Create a surface from the blended image
    blended_surface = pygame.surfarray.make_surface(blended_image)
    screen.blit(pygame.transform.scale(blended_surface, (width, height)), (0, 0))

    pygame.display.flip()

    clock.tick(fps)

pygame.quit()
