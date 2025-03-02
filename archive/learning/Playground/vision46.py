#!/usr/bin/env python3
#
# vision46.py

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import random

# Screen dimensions
width, height = 800, 600

# Particle class
class Particle:
    def __init__(self):
        self.position = [random.random()*width, random.random()*height]
        self.velocity = [random.random()*0.1 - 0.05, random.random()*0.1 - 0.05]
        self.lifespan = random.random() * 100 + 100

# Initialize Pygame
pygame.init()
display = (width, height)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Create particles
particles = [Particle() for _ in range(10000)]

# Invisible contours
contours = [[random.random()*width, random.random()*height] for _ in range(1000)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update particles
    for particle in particles:
        particle.position[0] += particle.velocity[0]
        particle.position[1] += particle.velocity[1]
        particle.lifespan -= 1
        if particle.lifespan <= 0:
            particles.remove(particle)
            particles.append(Particle())
        for contour in contours:
            if abs(particle.position[0] - contour[0]) < 10 and abs(particle.position[1] - contour[1]) < 10:
                particle.velocity[0] *= 1 
                particle.velocity[1] *= 1  

    # Render particles
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glBegin(GL_POINTS)
    for particle in particles:
        glColor3f(1, 1, 1)
        glVertex3fv((particle.position[0]/width*6-3, particle.position[1]/height*6-3, 0))
    glEnd()

    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
