#!/usr/bin/env python3
#
# seis_pygame.py

import pygame
import serial
import sys
import random

# Set the serial port and baud rate
serialPort = "/dev/cu.usbserial-210"
baudRate = 115200

# Open the serial port
ser = serial.Serial(serialPort, baudRate)

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 640, 480
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
MAX_POINTS = 10000 

# Create the display surface
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Create a list to hold our data points
data = []

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Add a new data point
    if len(data) < MAX_POINTS:
        data.insert(0, random.randint(0, HEIGHT))
    else:
        data.pop()
        data.insert(0, random.randint(0, HEIGHT))

    # Clear the screen
    screen.fill(WHITE)

    # Draw the line graph
    for i in range(1, len(data)):
        pygame.draw.line(screen, BLUE, (WIDTH - i, data[i - 1]), (WIDTH - (i + 1), data[i]))

    # Flip the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()

