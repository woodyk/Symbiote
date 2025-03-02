#!/usr/bin/env python3
#
# vision2.py

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# Screen size
width, height = 800, 600

# Initialize Pygame
pygame.init()
pygame.display.set_mode((width, height), DOUBLEBUF|OPENGL)

# Set the perspective of the 3D scene
gluPerspective(45, (width/height), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    # Generate random pixel values for grayscale static
    gray_data = np.random.randint(0, 256, (100,), dtype=np.uint8)

    # Use the grayscale values to create a 3D point cloud
    glBegin(GL_POINTS)
    for x in range(gray_data.shape[0]):
        glColor3f(gray_data[x]/255, gray_data[x]/255, gray_data[x]/255)
        glVertex3f((x % 10) - 5, (x // 10) - 5, (gray_data[x]) / 255 * 10 - 5)
    glEnd()

    # Generate random pixel values for RGB static
    rgb_data = np.random.randint(0, 256, (100, 3), dtype=np.uint8)

    # Use the color values to create a 3D point cloud
    glBegin(GL_POINTS)
    for x in range(rgb_data.shape[0]):
        glColor3f(rgb_data[x, 0]/255, rgb_data[x, 1]/255, rgb_data[x, 2]/255)
        glVertex3f((x % 10) - 5, (x // 10) - 5, (rgb_data[x, 0] + rgb_data[x, 1] + rgb_data[x, 2]) / (3 * 255) * 10 - 5)
    glEnd()

    # Update the display
    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
