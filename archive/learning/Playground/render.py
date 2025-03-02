#!/usr/bin/env python3

import pygame
import serial
import sys
import numpy as np
import threading
import random

# Initialize Pygame
pygame.init()

# Set up some initial constants
WIDTH, HEIGHT = 640, 480
#WIDTH, HEIGHT = 320, 240 
DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

# Set up serial port
ser = serial.Serial('/dev/ttyACM0', 115200)

# Initialize clock and set up frames per second
clock = pygame.time.Clock()
FPS = 30 

def map_value(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

# Fast sigmoid approximation
def sigmoid(x):
    return x / (1 + np.abs(x))

# Initialize 2D tensor
tensor = np.zeros((WIDTH, HEIGHT))

def read_serial():
    global tensor
    while True:
        if ser.in_waiting:
            try:
                #value = ser.readline().strip().decode('utf-8', 'ignore') # decode bytes to string, ignoring errors
                value = int(value) # convert string to int
                value = max(0, min(1023, value))
                #tensor[:, -1] = sigmoid(value / 1023.0)
                tensor[:, -1] = random.uniform(0.0, 1.0) 
                tensor = np.roll(tensor, -1, axis=1)
            except ValueError:
                continue

threading.Thread(target=read_serial, daemon=True).start()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.size
            WIDTH = max(420, WIDTH) # enforce minimum width
            HEIGHT = max(420, HEIGHT) # enforce minimum height
            DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            tensor = np.zeros((WIDTH, HEIGHT)) # Reset the tensor size

    # Create pixel array
    pixels = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)

    # Map tensor values to pixel colors
    pixels = (tensor * 255).astype(np.uint8)
    pixels = np.repeat(pixels[:, :, np.newaxis], 3, axis=2)

    # Create Pygame surface from pixel array
    image_surface = pygame.surfarray.make_surface(pixels)

    # Blit surface to screen
    DISPLAYSURF.blit(image_surface, (0, 0))

    # Update display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(FPS)

