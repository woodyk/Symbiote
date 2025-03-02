#!/usr/bin/env python3
#
# symbios_renderer.py

import pygame
import numpy as np

class SymbiosRenderer:
    def __init__(self, width=640, height=480):
        pygame.init()
        self.width = width
        self.height = height
        self.display_surf = pygame.display.set_mode((self.width, self.height))

    def render_frame(self, frame_data):
        # Create a Pygame surface from the frame data
        image_surface = pygame.surfarray.make_surface(frame_data)

        # Blit the image surface onto the display surface
        self.display_surf.blit(image_surface, (0, 0))

        # Update the display
        pygame.display.flip()

    def render_shape(self, shape_data):
        # Parse the shape data
        shape = shape_pb2.Shape()
        shape.ParseFromString(shape_data)

        # Render the shape
        if shape.type == shape_pb2.Shape.SQUARE:
            # Render a square
            pass
        elif shape.type == shape_pb2.Shape.CIRCLE:
            # Render a circle
            pass
        elif shape.type == shape_pb2.Shape.LINE:
            # Render a line
            pass

    def main_loop(self, frame_generator):
        # Main Pygame loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get the next frame from the generator and render it
            frame_data = next(frame_generator)
            self.render_frame(frame_data)

        pygame.quit()

import math

# Define the vertices of the cube
vertices = np.array([
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1]
]) * 100  # Scale the cube

# Define the edges of the cube
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# Define a function to rotate the cube
def rotate_cube(angle, vertices):
    rotation_matrix = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(vertices, rotation_matrix)

# Create an instance of the SymbiosRenderer class
renderer = SymbiosRenderer()

# Define a generator that yields the frame data for each frame
def frame_generator():
    angle = 0
    while True:
        # Clear the display surface
        renderer.display_surf.fill((0, 0, 0))

        # Rotate the cube
        rotated_vertices = rotate_cube(angle, vertices)

        # Draw the edges of the cube
        for edge in edges:
            x1, y1, _ = rotated_vertices[edge[0]] + np.array([renderer.width // 2, renderer.height // 2, 0])
            x2, y2, _ = rotated_vertices[edge[1]] + np.array([renderer.width // 2, renderer.height // 2, 0])
            pygame.draw.line(renderer.display_surf, (255, 255, 255), (x1, y1), (x2, y2))

        # Update the display
        pygame.display.flip()

        # Increment the angle
        angle += 0.01

        # Yield a dummy frame data array (not used in this example)
        yield np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)

# Start the main loop of the renderer, passing the frame generator
renderer.main_loop(frame_generator())
