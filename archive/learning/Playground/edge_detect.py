#!/usr/bin/env python3
#
# edge_detect.py

import cv2
import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Get the size of the screen
infoObject = pygame.display.Info()
screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h))

# Create a surface we can draw on
surface = pygame.Surface((infoObject.current_w, infoObject.current_h))

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image from OpenCV BGR format to Pygame RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply edge detection
    edges = cv2.Canny(frame,100,200)

    # Convert binary image to 3-channel image
    edges_3d = np.stack((edges,)*3, axis=-1)

    # Convert the image to a format Pygame can use
    frame = pygame.image.frombuffer(edges_3d.tostring(), edges_3d.shape[1::-1], "RGB")

    # Draw the image to the screen
    surface.blit(frame, (0,0))
    screen.blit(surface, (0,0))

    # Update the display
    pygame.display.flip()

    # Check for quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            cv2.destroyAllWindows()
            quit()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
