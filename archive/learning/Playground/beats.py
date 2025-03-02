#!/usr/bin/env python3
#
# beats.py

import pygame
from pygame.locals import *

# Initialize pygame
pygame.init()

# Define the screen size
screen_size = (800, 600)

# Create the screen
screen = pygame.display.set_mode(screen_size)

# Define the grid size
grid_size = (8, 8)

# Define the square size
square_size = (50, 50)

# Define the colors
white = (255, 255, 255)
black = (0, 0, 0)

# Define the sounds
sounds = ['sound1.wav', 'sound2.wav', 'sound3.wav', 'sound4.wav']

# Load the sounds
sound_objects = [pygame.mixer.Sound(sound) for sound in sounds]

# Create a 2D array to store the state of the squares
squares = [[0 for _ in range(grid_size[0])] for _ in range(grid_size[1])]

# Main loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            # Get the mouse position
            mouse_pos = pygame.mouse.get_pos()
            
            # Calculate the grid position
            grid_pos = (mouse_pos[0] // square_size[0], mouse_pos[1] // square_size[1])
            
            # Toggle the state of the square
            squares[grid_pos[0]][grid_pos[1]] = 1 - squares[grid_pos[0]][grid_pos[1]]
            
            # Play the corresponding sound
            sound_objects[grid_pos[0]].play()
    
    # Drawing
    screen.fill(black)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if squares[i][j] == 1:
                pygame.draw.rect(screen, white, (i * square_size[0], j * square_size[1], square_size[0], square_size[1]))
    pygame.display.flip()

# Quit pygame
pygame.quit()
