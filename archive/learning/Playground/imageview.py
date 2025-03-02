#!/usr/bin/env python3
#
# imageview.py

import pygame
import requests
import io
import os
import re
from pygame.locals import *

class ImageViewer:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.index = 0
        self.zoom_level = 1

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)

    def load_image(self, path):
        if os.path.isfile(path):
            return pygame.image.load(path)
        elif os.path.isdir(path):
            self.browse_directory(path)
        elif re.search(r'^http+S:\/\/', path):
            try:
                response = requests.get(path)
                response.raise_for_status()
                image_file = io.BytesIO(response.content)
                return pygame.image.load(image_file)
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                return None
        else:
            return None

    def draw_image(self):
        image = self.load_image(self.image_paths[self.index])
        if image is not None:
            image = pygame.transform.scale(image, (int(image.get_width() * self.zoom_level), int(image.get_height() * self.zoom_level)))
            self.screen.blit(image, (0, 0))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.index = (self.index - 1) % len(self.image_paths)
                    elif event.key == pygame.K_RIGHT:
                        self.index = (self.index + 1) % len(self.image_paths)
                    elif event.key == pygame.K_UP:
                        self.zoom_level *= 1.1
                    elif event.key == pygame.K_DOWN:
                        self.zoom_level /= 1.1
                    elif event.key == pygame.K_ESCAPE:
                        # TODO: Go back to the previous view
                        pass
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            self.screen.fill((0, 0, 0))
            self.draw_image()
            pygame.display.flip()

        pygame.quit()

    def browse_directory(self, directory_path):
        if os.path.isdir(directory_path):
            for filename in os.listdir(directory_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    self.image_paths.append(os.path.join(directory_path, filename))
            self.run()
        else:
            print(f"Error: {directory_path} is not a directory")

'''
# List of image URLs or local file paths
image_paths = [
    "https://example.com/image1.png",
    "https://example.com/image2.png",
    "/path/to/local/image3.png",
    "/path/to/local/image4.png",
]

# Create an ImageViewer instance
viewer = ImageViewer(image_paths)

# Run the image viewer
viewer.run()
'''

image_paths = [ "~/Pictures/IMG_4451.jpg" ]
viewer = ImageViewer(image_paths)
viewer.run()
