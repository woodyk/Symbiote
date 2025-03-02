#!/usr/bin/env python3

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random

# Particle class
class Particle:
    def __init__(self):
        self.position = [random.uniform(-1.0, 1.0) for _ in range(3)]
        self.velocity = [random.uniform(-0.02, 0.02) for _ in range(3)]

    def move(self):
        for i in range(3):
            self.position[i] += self.velocity[i]

            if abs(self.position[i]) > 1:
                self.velocity[i] *= -1

# Initialize particles
particles = [Particle() for _ in range(1000)]

# Main function
def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glBegin(GL_POINTS)
        for particle in particles:
            particle.move()
            glVertex3fv(particle.position)
        glEnd()

        pygame.display.flip()
        pygame.time.wait(10)

# Run the main function
if __name__ == "__main__":
    main()

