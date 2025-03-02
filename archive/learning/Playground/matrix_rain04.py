#!/usr/bin/env python3
#
# matrix_rain04.py

import pygame
import random

# Window size
WIDTH, HEIGHT = 800, 600

# Parameters
FONT_SIZE = 15
FONT_NAME = "couriernew"
FONT_COLOR = (0, 255, 0)
BACKGROUND_COLOR = (0, 0, 0)
SPEED = 2

# Initialize Pygame
pygame.init()

# Set up some necessary values
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
characters = [chr(int("33" + str(i), 16)) for i in range(94)]
column_count = WIDTH // FONT_SIZE

# Create a 2D list that will represent the grid of raindrops
rain_grid = [[None for _ in range(HEIGHT // FONT_SIZE)] for _ in range(column_count)]

# Main loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Animation
    for i in range(column_count):
        if rain_grid[i][0] is None and random.randrange(100) < 2:
            rain_grid[i][0] = characters[random.randrange(94)]
        for j in range(1, HEIGHT // FONT_SIZE):
            if rain_grid[i][j - 1] is not None:
                rain_grid[i][j] = rain_grid[i][j - 1]
                rain_grid[i][j - 1] = None

    # Drawing
    screen.fill(BACKGROUND_COLOR)
    for i in range(column_count):
        for j in range(HEIGHT // FONT_SIZE):
            if rain_grid[i][j] is not None:
                screen.blit(font.render(rain_grid[i][j], True, FONT_COLOR), (i * FONT_SIZE, j * FONT_SIZE))
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.wait(1000 // SPEED)

# Clean up
pygame.quit()
