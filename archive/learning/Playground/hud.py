#!/usr/bin/env python3
#
# hud.py

import cv2
import sys
import pygame
import pygame_gui
import numpy as np
from collections import deque

# Initialize Pygame
pygame.init()

# Set the display resolution
display_width = 800
display_height = 600
screen = pygame.display.set_mode((display_width, display_height))

# Create a surface we can draw on
surface = pygame.Surface((display_width, display_height))

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the Haar cascade xml file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set the edge detection sensitivity
min_val = 50
max_val = 150

# Set the control flags
edge_detection = True
face_detection = True
camera_output = True
filter_only = False
motion_detection = False

# Set the menu flag
menu = False

# Set the font for the help menu
font = pygame.font.SysFont('couriernew', 20)

# Initialize the previous frame for motion detection
prev_frame = None

# Initialize the motion history for fading blur effect
motion_history = deque(maxlen=5)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to match the display resolution
    frame = cv2.resize(frame, (display_width, display_height))

    # Convert the image from OpenCV BGR format to Pygame RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Store the original frame before any blurring is done
    original_frame = frame.copy()

    # Apply edge detection
    if edge_detection:
        edges = cv2.Canny(frame, min_val, max_val)
        # Convert binary image to 3-channel image
        edges_3d = np.stack((edges,)*3, axis=-1)
        frame[edges != 0] = [0, 255, 0] # Change color of the edges to red

    # Apply motion detection
    if motion_detection:
        if prev_frame is not None:
            # Calculate the absolute difference between the current frame and the previous frame
            diff = cv2.absdiff(prev_frame, original_frame)
            # If the difference is above a threshold, consider it as motion
            motion = diff > 55 
            # Convert the motion mask to a grayscale image
            motion_gray = cv2.cvtColor(motion.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            # Add the current motion mask to the motion history
            motion_history.appendleft(motion_gray)
            # Apply a fading highlight effect based on the motion history
            for i, motion_gray in enumerate(motion_history):
                highlight_intensity = 255 * (1.0 - 0.1 * i)
                frame[motion_gray != 0] = [highlight_intensity, 0, 0]  # Change color of the motion areas to red
        # Update the previous frame with the original frame
        prev_frame = original_frame.copy()

    # Convert the image back to BGR format for face detection
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Perform face detection
    if face_detection:
        faces = face_cascade.detectMultiScale(frame_bgr, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert the image back to RGB format for Pygame
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if filter_only:
        frame = pygame.image.frombuffer(edges_3d.tobytes(), edges_3d.shape[1::-1], "RGB")
    else:
        frame = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")

    # Draw the image to the screen
    if camera_output:
        surface.blit(frame, (0,0))

    screen.blit(surface, (0,0))

    # Display the menu
    if menu:
        s = pygame.Surface((display_width,display_height)) # the size of your rect
        s.set_alpha(128) # alpha level
        s.fill((0,0,0)) # this fills the entire surface
        screen.blit(s, (0,0)) # (0,0) are the top-left coordinates

        # Define the help text
        help_text = [
            "Controls:",
            "1: Toggle edge detection",
            "2: Toggle face detection",
            "3: Toggle camera output",
            "4: Toggle filter only mode",
            "5: Toggle motion detection",
            "E: Increase edge detection sensitivity",
            "R: Decrease edge detection sensitivity",
            "Q: Quit / Exit"
        ]

        # Render the help text
        for i, line in enumerate(help_text):
            text_surface = font.render(line, True, (0, 255, 0))
            screen.blit(text_surface, (10, 10 + 30 * i))

    # Update the display
    pygame.display.flip()

    # Check for quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            cv2.destroyAllWindows()
            quit()

        # Check for keypress events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                menu = not menu
            elif event.key == pygame.K_1:
                edge_detection = not edge_detection
            elif event.key == pygame.K_2:
                face_detection = not face_detection
            elif event.key == pygame.K_3:
                camera_output = not camera_output
            elif event.key == pygame.K_4:
                filter_only = not filter_only
            elif event.key == pygame.K_5:
                motion_detection = not motion_detection
            elif event.key == pygame.K_e:
                min_val = max(0, min_val - 10)
                max_val = max(0, max_val - 10)
            elif event.key == pygame.K_r:
                min_val = min(255, min_val + 10)
                max_val = min(255, max_val + 10)
            elif event.key == pygame.K_q:
                sys.exit()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

