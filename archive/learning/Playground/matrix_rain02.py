#!/usr/bin/env python3
#
# matrix_rain02.py

import random
import pygame
import pygame.freetype

# Matrix parameters
matrix = 0
num_streams = 100
speed = .01
min_speed = 0.01
max_speed = 5.0
font_size = 12 
font_color = (255, 255, 255, 255) # White
font_decay_color = (0, 255, 0, 255) # Green
font_intensity_decay = 10 
fade_intensity = 10 # The higher this value, the faster the trails will fade
new_stream = True

# Initialize Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.freetype.SysFont('Courier', font_size)

# Get screen resolution
info = pygame.display.Info()
width, height = info.current_w, info.current_h

# Frames per second
FPS = 30

# Calculate number of columns
num_columns = width // font_size

# Create streams
new_streams = []

def create_streams(matrix):
    streams = []
    if matrix == 1:
        for _ in range(num_columns):
            speed = random.uniform(min_speed, max_speed)
            x = _ * font_size
            y = random.randint(0, height) # Random value within the height of the screen
            streams.append((speed, x, y))
    elif matrix == 0:
        for _ in range(num_columns):
            for __ in range(height // font_size):
                speed = random.uniform(min_speed, max_speed)
                x = _ * font_size
                y = __ * font_size
                color = list(font_color) # Make a copy of the color
                streams.append((speed, x, y, color))

    return streams

# Help menu
help_menu = False

def display_help_menu():
    # Create a semi-transparent surface
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((0, 0, 0, 128)) # Semi-transparent black

    # Create a font object
    font = pygame.freetype.SysFont('Courier', 16)

    # Define the help text
    help_text = [
        "Help Menu:",
        "UP: Increase speed",
        "DOWN: Decrease speed",
        "LEFT: Increase persistence",
        "RIGHT: Decrease persistence",
        "+: Increase font size",
        "-: Decrease font size",
        "H: Show/hide this help menu",
        "I: Increase fade intensity",
        "D: Decrease fade intensity",
        "N: Increase density",
        "M: Decrease density",
    ]

    # Render the help text
    for i, line in enumerate(help_text):
        font.render_to(surface, (10, 10 + 20*i), line, (255, 255, 255)) # White text

    return surface

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                max_speed = min(10, max_speed + 0.01)
            elif event.key == pygame.K_DOWN:
                max_speed = max(0.5, max_speed - 0.01)
            elif event.key == pygame.K_LEFT:
                font_intensity_decay = max(0.01, font_intensity_decay - 0.01)
            elif event.key == pygame.K_RIGHT:
                font_intensity_decay = min(10, font_intensity_decay + 0.01)
            elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                font_size += 2
                font = pygame.freetype.SysFont('Courier', font_size)
            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                font_size = max(2, font_size - 2)
                font = pygame.freetype.SysFont('Courier', font_size)
            elif event.key == pygame.K_h:
                help_menu = not help_menu # Toggle help menu
            elif event.key == pygame.K_EQUALS:
                font_size += 2
                font = pygame.freetype.SysFont('Courier', font_size)
            elif event.key == pygame.K_MINUS:
                font_size = max(2, font_size - 2)
                font = pygame.freetype.SysFont('Courier', font_size)
            elif event.key == pygame.K_n:
                new_stream = True
                if matrix >= 1:
                    matrix = 0
                else:
                    matrix = matrix + 1
            elif event.key == pygame.K_i:
                fade_intensity = min(10, fade_intensity + 0.01)
            elif event.key == pygame.K_d:
                fade_intensity = max(0.01, fade_intensity - 0.01)
            elif event.key == pygame.K_v:
                num_columns += 10
            elif event.key == pygame.K_b:
                num_columns = max(10, num_columns - 10)

    if new_stream:
        streams = create_streams(matrix)
        new_stream = False

    # Clear screen
    screen.fill((0, 0, 0))

    # Clear screen with a semi-transparent black rectangle
    fade_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    fade_surface.fill((0, 0, 0, fade_intensity))
    screen.blit(fade_surface, (0, 0))

    # Update and draw streams
    new_streams = []
    if matrix == 0:
        for speed, x, y, color in streams:
            # Update color
            color[3] = max(0, color[3] - fade_intensity) # Reduce alpha component

            # Draw character
            char = chr(random.randint(0x20, 0x7E)) # ASCII characters
            font.render_to(screen, (x, int(y)), char, color)

            # Update position
            y += speed
            if y > height:
                y = 0
                color = list(font_color) # Reset color
            new_streams.append((speed, x, y, color))
        streams = new_streams
    elif matrix == 1:
        for speed, x, y in streams:
            # Draw characters
            for i in range(int(y), height, font_size):
                intensity = min(1, max(0, (i - y) / height))
                color = [int(c * intensity) for c in font_color]
                decay_color = [int(c * (1 - intensity)) for c in font_decay_color]
                final_color = [c1 + c2 for c1, c2 in zip(color, decay_color)]
                char = chr(random.randint(0x20, 0x7E)) # ASCII characters
                font.render_to(screen, (x, i), char, final_color)

            # Update stream
            y += speed
            if y > height:
                y = random.randint(-height, 0)
            new_streams.append((speed, x, int(y)))
        streams = new_streams

    # Display help menu
    if help_menu:
        help_surface = display_help_menu()
        screen.blit(help_surface, (0, 0))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
