#!/usr/bin/env python3
#
# vision30.py

#!/usr/bin/env python3
#
# vision30.py

import pygame
import numpy as np

# Define the size of the image and frames per second
width, height = 800, 600
fps = 60

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Initialize the channel number
channel = 0

# Initialize the font
font = pygame.font.SysFont('courier', 20)

# Define the controls
controls = ['UP: Increase speed', 'DOWN: Decrease speed', 'LEFT: Previous channel', 'RIGHT: Next channel']

# Initialize a list to store the snowflakes
snowflakes = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(500)]

# Set the initial speed of the snowflakes
speed = 1

# Initialize a list to store the shapes
shapes = []

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                speed = min(10, speed + 1)
            elif event.key == pygame.K_DOWN:
                speed = max(1, speed - 1)
            elif event.key == pygame.K_LEFT:
                channel = max(0, channel - 1)
            elif event.key == pygame.K_RIGHT:
                channel = min(56, channel + 1)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Update the position of the snowflakes
    for i, (x, y) in enumerate(snowflakes):
        x += np.random.randint(-1, 2)  # shift direction left or right
        y += speed  # fall down
        if y > height:  # if the snowflake has fallen past the bottom
            y = 0  # move it back to the top
            x = np.random.randint(0, width)  # and give it a new x position
        snowflakes[i] = (x, y)

        # Draw the snowflake
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 1)

    # Generate random shapes
    if np.random.random() < 0.01:  # 1% chance each frame
        shapes.append((np.random.randint(0, width), np.random.randint(0, height), np.random.randint(10, 50)))

    # Display the channel number
    channel_text = font.render(str(channel), True, (0, 255, 0))
    screen.blit(channel_text, (width - channel_text.get_width() - 10, height - channel_text.get_height() - 10))

    # Display the controls
    for i, control in enumerate(controls):
        control_text = font.render(control, True, (255, 255, 255))
        screen.blit(control_text, (10, height - (len(controls) - i) * control_text.get_height()))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()

