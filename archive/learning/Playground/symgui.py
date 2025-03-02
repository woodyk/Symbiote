#!/usr/bin/env python3
#
# symbgui.py

import os
import sys
import pygame
import pygame_gui
import symbiote.core as core
import symbiote.chat as chat

class StdoutRedirect:
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, s):
        self.textbox.html_text += '<br>' + s
        self.textbox.rebuild()

    def flush(self):
        pass

    def isatty(self):
        return False

class SymbioteGUI:
    def __init__(self):
        current_path = os.getcwd()

        pygame.init()

        self.screen_width = 1024 
        self.screen_height = 768 
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)

        self.background = pygame.Surface((self.screen_width, self.screen_height))
        self.background.fill(pygame.Color('#000000'))

        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height))

        '''
        self.chat_history = pygame_gui.elements.ui_text_box.UITextBox(
            '',
            relative_rect=pygame.Rect((250, 20), (700, 500)),
            manager=self.manager
        )
        '''
        # Create a scrolling container
        self.scrolling_container = pygame_gui.elements.UIScrollingContainer(
            relative_rect=pygame.Rect((250, 20), (700, 500)),
            manager=self.manager
        )

        # Create the text box inside the scrolling container
        self.chat_history = TerminalLikeTextBox(
            relative_rect=pygame.Rect((0, 0), (700, 500)),
            html_text='',
            manager=self.manager,
            container=self.scrolling_container  # Set the container to the scrolling container
        )

        self.chat_input = pygame_gui.elements.ui_text_entry_line.UITextEntryLine(
            relative_rect=pygame.Rect((250, 530), (700, 50)),
            manager=self.manager
        )

        self.clock = pygame.time.Clock()

        self.schat = chat.symchat(working_directory=current_path, debug=False)

        # Redirect stdout and stderr to the chat_history text box
        sys.stdout = self.chat_history
        sys.stderr = self.chat_history

    def run(self):
        running = True

        while running:
            time_delta = self.clock.tick(60)/1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                self.manager.process_events(event)

                if event.type == pygame.USEREVENT:
                    if event.user_type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                        user_input = self.chat_input.get_text()
                        response = self.schat.chat(user_input=user_input, run=True)
                        if isinstance(response, list):
                            chat_response = response[0].pop()
                            returned = chat_response['content']
                        else:
                            returned = "OK"

                        self.chat_history.html_text += '<br>User: ' + user_input
                        self.chat_history.html_text += '<br>Symbiote: ' + returned
                        self.chat_history.rebuild()
                        self.chat_input.set_text('')

            self.manager.update(time_delta)

            self.screen.blit(self.background, (0, 0))
            self.manager.draw_ui(self.screen)

            pygame.display.update()

        pygame.quit()

class TerminalLikeTextBox(pygame_gui.elements.UITextBox):
    def __init__(self, html_text, relative_rect, manager, container=None):
        super().__init__(html_text, relative_rect, manager, container=container)

    def write(self, s):
        # Append the string to the text box
        self.html_text += '<br>' + s
        self.rebuild()

        # Scroll to the bottom
        max_scroll_value = self.vertical_scroll_bar.scrollable_height - self.vertical_scroll_bar.visible_percentage
        self.vertical_scroll_bar.scroll_position = max_scroll_value
        self.vertical_scroll_bar.start_percentage = max_scroll_value

    def flush(self):
        pass

    def isatty(self):
        return False

    def set_scroll_position(self, position):
        # Call the set_scroll_position method of the superclass
        super().set_scroll_position(position)

class TerminalLikeTextEntryLine(pygame_gui.elements.UITextEntryLine):
    def __init__(self, rect, manager):
        super().__init__(rect, manager)
        self.command_history = []
        self.current_command_index = -1

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                # Show the previous command
                if self.current_command_index < len(self.command_history) - 1:
                    self.current_command_index += 1
                    self.set_text(self.command_history[self.current_command_index])
                return True
            elif event.key == pygame.K_DOWN:
                # Show the next command
                if self.current_command_index > 0:
                    self.current_command_index -= 1
                    self.set_text(self.command_history[self.current_command_index])
                return True
        elif event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            # Add the command to the history
            self.command_history.insert(0, self.get_text())
            self.current_command_index = -1
        return super().process_event(event)

if __name__ == "__main__":
    gui = SymbioteGUI()
    gui.run()
