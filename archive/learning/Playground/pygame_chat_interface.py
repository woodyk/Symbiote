#!/usr/bin/env python3
#
# tt.py

import pygame
import pygame.freetype

class ChatInterface:
    def __init__(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.freetype.Font(None, 24)
        self.input_box = pygame.Rect(100, 100, 140, 32)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.active = False
        self.text = ''
        self.done = False
        self.messages = []

    def draw(self):
        self.screen.fill((30, 30, 30))
        txt_surface = self.font.render(self.text, self.color)
        width = max(200, txt_surface[1].width+10)
        self.input_box.w = width
        self.font.render_to(self.screen, (self.input_box.x+5, self.input_box.y+5), self.text, self.color)
        pygame.draw.rect(self.screen, self.color, self.input_box, 2)

        y = 5
        for message in self.messages:
            self.font.render_to(self.screen, (5, y), message, pygame.Color('white'))
            y += 20

        pygame.display.flip()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.input_box.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    response = self.send_message(self.text)
                    self.messages.append('You: ' + self.text)
                    self.messages.append('Bot: ' + response)
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode

    def run(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                self.handle_event(event)
            self.draw()

    def send_message(self, message):
        # Replace this with your chatbot's response method
        return "I received your message: " + message

if __name__ == "__main__":
    chat = ChatInterface(800, 600)
    chat.run()
