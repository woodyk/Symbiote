#!/usr/bin/env python3
#
# smudge.py

import pygame
import random

# Initialize Pygame
pygame.init()

# Set the dimensions of the screen
screen = pygame.display.set_mode((800, 600))

# Set the title of the window
pygame.display.set_caption("Monitor Smudges")

# Define the smudge class
class Smudge:
    def __init__(self):
        self.x = random.randint(0, 800)
        self.y = random.randint(0, 600)
        self.width = random.randint(1, 50)
        self.height = random.randint(1, 10)
        self.color = (100, 100, 100, random.randint(50, 255))  # Semi-transparent grey
        self.orientation = random.choice(['horizontal', 'vertical'])

    def draw(self):
        s = pygame.Surface((800,600))  # the size of your rect
        s.set_alpha(128)              # alpha level
        s.fill((255,255,255))         # this fills the entire surface
        if self.orientation == 'horizontal':
            pygame.draw.line(s, self.color, (self.x, self.y), (self.x + self.width, self.y), self.height)
        else:
            pygame.draw.line(s, self.color, (self.x, self.y), (self.x, self.y + self.height), self.width)
        screen.blit(s, (0,0))    # (0,0) are the top-left coordinates

    def update(self):
        self.width += random.randint(-1, 1)
        self.height += random.randint(-1, 1)

# Create a list of smudges
smudges = [Smudge() for _ in range(10)]

# Game loop
running = True
while running:
    # Fill the screen with black
    screen.fill((0, 0, 0))

    # Draw and update smudges
    for smudge in smudges:
        smudge.draw()
        smudge.update()

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                print("Controls:")
                print("  - Increase size: W")
                print("  - Decrease size: Z")
                print("  - Increase number of smudges: D")
                print("  - Decrease number of smudges: X")

            elif event.key == pygame.K_w:
                for smudge in smudges:
                    smudge.width += 1
                    smudge.height += 1
            elif event.key == pygame.K_z:
                for smudge in smudges:
                    smudge.width -= 1
                    smudge.height -= 1
            elif event.key == pygame.K_d:
                smudges.append(Smudge())
            elif event.key == pygame.K_x:
                if smudges:
                    smudges.pop()

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()

