#!/usr/bin/env python3
#
# vision101.py

#!/usr/bin/env python3

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import cv2

# Initialize Pygame 
pygame.init()
display = (800,600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
pygame.display.set_caption('OpenGL Pygame Panel')

# Define the size of the image and frames per second
width, height = 800, 600

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

def main():
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    noise_level = min(256, noise_level + 10)
                elif event.key == pygame.K_DOWN:
                    noise_level = max(1, noise_level - 10) # Set the lower limit to 1
                
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
        
        pygame.display.flip()
        pygame.time.wait(10)
    pygame.quit()

main()
