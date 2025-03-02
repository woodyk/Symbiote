#!/usr/bin/env python3
#
# tt.py

import pygame
import random
import modules.graph as graph

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 640, 480
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
MAX_POINTS = 1000 

# Create the display surface
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Create a Graph object
graph = graph.Graph(screen, WIDTH, HEIGHT, MAX_POINTS, BLUE)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Add a new data point
    graph.add_data(random.randint(0, HEIGHT))

    # Clear the screen
    screen.fill(WHITE)

    # Draw the graph
    graph.draw_scatter()  # or graph.draw_bar() or graph.draw_area()

    # Flip the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
