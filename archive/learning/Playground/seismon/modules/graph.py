#!/usr/bin/env python3
#
# graph.py

import pygame
import random

class Graph:
    def __init__(self, screen, width, height, max_points, color):
        self.screen = screen
        self.width = width
        self.height = height
        self.max_points = max_points
        self.color = color
        self.data = []

    def add_data(self, value):
        if len(self.data) < self.max_points:
            self.data.insert(0, value)
        else:
            self.data.pop()
            self.data.insert(0, value)

    def draw_line(self):
        for i in range(1, len(self.data)):
            pygame.draw.line(self.screen, self.color, (self.width - i, self.data[i - 1]), (self.width - (i + 1), self.data[i]))

    def draw_bar(self):
        bar_width = self.width // self.max_points
        for i in range(len(self.data)):
            pygame.draw.rect(self.screen, self.color, pygame.Rect((self.width - (i + 1) * bar_width, self.height - self.data[i]), (bar_width, self.data[i])))

    def draw_area(self):
        for i in range(1, len(self.data)):
            pygame.draw.polygon(self.screen, self.color, [(self.width - i, self.height), (self.width - i, self.height - self.data[i]), (self.width - (i + 1), self.height - self.data[i - 1]), (self.width - (i + 1), self.height)])

    def draw_scatter(self):
        point_radius = 3  # You can adjust this value as needed
        for i in range(len(self.data)):
            pygame.draw.circle(self.screen, self.color, (self.width - i, self.height - self.data[i]), point_radius)
