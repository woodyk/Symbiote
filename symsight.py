#!/usr/bin/env python3
#
# matrix_rain.py

import sys
import os
import time
import json
import random
import pygame
import pygame.freetype
import pytesseract
import numpy as np
import cv2

class ColorConverter:
    def __init__(self):
        # Manually define a dictionary of CSS3 colors in hexadecimal format
        self.colors = {
            'black': '#000000',
            'white': '#FFFFFF',
            'red': '#FF0000',
            'green': '#008000',
            'blue': '#0000FF',
            'yellow': '#FFFF00',
            'cyan': '#00FFFF',
            'magenta': '#FF00FF',
            # Add more as needed
            'neon_green': '#39ff14',
            'neon_yellow': '#ffff00',
            'neon_red': '#ff0033',
            'neon_blue': '#4d4dff'
        }

    def get_color_names(self, format='rgba'):
        color_dict = {}
        for name, hex_value in self.colors.items():
            rgb = self.hex_to_rgb(hex_value)
            if format == 'rgba':
                color_dict[name] = (*rgb, 255)
            else:
                color_dict[name] = rgb
        return color_dict

    def rgb_to_hex(self, rgb):
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Instantiate ColorConverter and get color names
color_converter = ColorConverter()
colors = color_converter.get_color_names()

# Matrix parameters and settings (remaining script as provided)
contour_detection = False
color_cycle = False
matrix = 0
num_streams = 10000
speed = .01
min_speed = 0.01
max_speed = 5.0
font_size = 12
default_font_color = 'white'
font_color = default_font_color
default_decay_color = 'green'
font_decay_color = default_decay_color
default_background_color = 'black'
background_color = default_background_color
fade_intensity = .1
random_fade = False
new_stream = True
persistent_dot = False
font = 'Courier'

setting_message = None
setting_message_expiration = 0
ocr_enabled = False
static_overlay = False

# Initialize Pygame
pygame.init()
width, height = 1024, 768
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
clock = pygame.time.Clock()
font = pygame.freetype.SysFont('Courier', font_size)
FPS = 30

# New variable for toggling edge detection
edge_detection = False
edge_fps = 30
edge_decay = 60
edge_detection_min = 50
edge_detection_max = 150
edge_surface = pygame.Surface((width, height), pygame.SRCALPHA)

# Create streams
new_streams = []
current_resolution = 0

def save_settings():
    print(f"Settings saved.")
    settings = {
        'max_speed': max_speed,
        'font_size': font_size,
        'fade_intensity': fade_intensity,
        'num_streams': num_streams,
        'persistent_dot': persistent_dot,
        'font_color': font_color,
        'font_decay_color': font_decay_color,
        'width': width,
        'height': height,
    }
    home = os.environ['HOME']
    with open(f'{home}/.symbiote/vison_settings.json', 'w') as f:
        json.dump(settings, f)

def load_settings():
    global max_speed, font_size, fade_intensity, num_streams, persistent_dot, font_color, font_decay_color, width, height
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
            max_speed = settings.get('max_speed', max_speed)
            font_size = settings.get('font_size', font_size)
            fade_intensity = settings.get('fade_intensity', fade_intensity)
            num_streams = settings.get('num_streams', num_streams)
            persistent_dot = settings.get('persistent_dot', persistent_dot)
            font_color = settings.get('font_color', font_color)
            font_decay_color = settings.get('font_decay_color', font_decay_color)
            width = settings.get('width', width)
            height = settings.get('height', height)
    except FileNotFoundError:
        pass  # It's okay if the file doesn't exist

#load_settings()

def detect_edges(surface):
    # Convert the Pygame surface to an OpenCV image
    cv_image = pygame.surfarray.array3d(surface)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny edge detection algorithm
    edges = cv2.Canny(gray, 50, 150)

    # Create a color mask for the edges
    color_mask = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    color_mask[edges != 0] = [0, 0, 255]  # Red color

    # Convert the color mask to a Pygame surface
    edges_surface = pygame.surfarray.make_surface(color_mask)

    return edges_surface

def draw_edges(surface):
    # Convert the Pygame surface to an OpenCV image
    cv_image = pygame.surfarray.array3d(surface)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny edge detection algorithm
    edges = cv2.Canny(gray, 50, 150)

    # Find the contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    #cv2.drawContours(cv_image, contours, -1, (0, 255, 0), 1)

    # Color the edges red
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_color[:,:,2] = edges  # Red channel

    # Convert the colored edges image to a Pygame surface
    edges_surface = pygame.surfarray.make_surface(edges_color)

    # Convert the OpenCV image back to a Pygame surface
    #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #edge_surface = pygame.surfarray.make_surface(cv_image)

    return edges_surface

# Modify the create_streams function to use the num_streams variable
def create_streams(matrix):
    streams = []
    if matrix == 0 or matrix == 2:
        for _ in range(num_streams):
            speed = random.uniform(min_speed, max_speed)
            x = random.randint(0, width)  # Random value within the width of the screen
            y = random.randint(0, height)  # Random value within the height of the screen
            color = list(colors[font_color])  # Make a copy of the color
            streams.append((speed, x, y, color))
    elif matrix == 1:
        for _ in range(num_streams):
            speed = random.uniform(min_speed, max_speed)
            x = random.randint(0, width)
            y = random.randint(0, height)
            color = list(colors[font_color])
            streams.append((speed, x, y))

    return streams

def detect_and_draw_contours(surface):
    # Get the dimensions of the display
    info = pygame.display.Info()
    width = info.current_w
    height = info.current_h

    # Convert the Pygame surface to an OpenCV image
    array = pygame.surfarray.array3d(surface)
    array = np.rot90(array)
    array = np.flipud(array)
    cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to highlight the contours
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find the contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new surface with the same size as the original
    contour_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # Draw the contours on the new surface
    for contour in contours:
        if len(contour) > 2:  # Only draw the contour if it has more than 2 points
            points = [tuple(point[0]) for point in contour]  # Convert arrays to tuples
            pygame.draw.polygon(contour_surface, (0, 255, 0, 128), points)

    return contour_surface

# Help menu
help_menu = False

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def display_help_menu():
    # Create a semi-transparent surface
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((0, 0, 0, 128))  # Semi-transparent black

    # Create a font object
    font = pygame.freetype.SysFont('Courier', 24)

    # Define the help text
    help_text = [
        "Help Menu:",
        "UP: Increase speed",
        "DOWN: Decrease speed",
        "+: Increase font size",
        "-: Decrease font size",
        "H: Show/hide this help menu",
        "I: Increase fade intensity",
        "D: Decrease fade intensity",
        "F: Toggle fullscreen",
        "K: Increase number of streams",
        "L: Decrease number of streams",
        "N: Cycle matrix vissual",
        "C: Toggle contour detection",
        "T: Toggle OCR detection",
        "R: Cycle through resolutions",
        "1: Toggle random fade intensity",
        "3: Toggle persistent dot", 
        "4: Cycle font color",
        "5: Cycle decay color",
        "6: Cycle background color",
        "8: Toggle edge detection",
        "9: Toggle static overlay",
        "S: Save settings",
        "Q: Quit / Exit",
    ]

    # Render the help text
    for i, line in enumerate(help_text):
        font.render_to(surface, (10, 10 + 20*i), line, colors['neon_green'])

    return surface

def generate_tv_static(width, height, alpha=128):
    static = np.random.randint(0, 255, (width, height, 3), dtype=np.uint8)

    # Convert the noise image to a Pygame surface
    static_surface = pygame.surfarray.make_surface(static)

    # Create a semi-transparent surface with the same size
    alpha_surface = pygame.Surface(static_surface.get_size(), pygame.SRCALPHA)
    alpha_surface.fill((0, 0, 0, alpha))

    # Blend the static with the semi-transparent surface
    static_surface.blit(alpha_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    return static_surface

def cycle_color(current):
    # Get a list of color names
    color_names = list(colors.keys())

    # Find the index of the current color
    current_index = color_names.index(current)

    # Get the next color in the list, wrapping around to the start if necessary
    next_index = (current_index + 1) % len(color_names)

    # Return the name and RGB values of the next color
    return color_names[next_index], colors[color_names[next_index]]

# Main loop
running = True
setting_message = False
count = 0
recog = 0
frame_counter = 0
while running:
    frame_counter += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            # Update the window size
            width, height = event.size
            # Recreate the streams to fit the new window size
            streams = create_streams(matrix)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                max_speed = min(50, max_speed + 0.1)
                setting_message = f"Fall rate: {max_speed}"
            elif event.key == pygame.K_DOWN:
                max_speed = max(0.1, max_speed - 0.1)
                setting_message = f"Fall rate: {max_speed}"
            elif event.key == pygame.K_h:
                help_menu = not help_menu  # Toggle help menu
            elif event.key == pygame.K_EQUALS:
                font_size += 2
                font = pygame.freetype.SysFont('Courier', font_size)
                setting_message = f"Font size: {font_size}"
            elif event.key == pygame.K_MINUS:
                font_size = max(2, font_size - 2)
                font = pygame.freetype.SysFont('Courier', font_size)
                setting_message = f"Font size: {font_size}"
            elif event.key == pygame.K_n:
                new_stream = True
                if matrix >= 3:
                    matrix = 0
                else:
                    matrix += 1
                if matrix == 1:
                    num_streams = 100
                if matrix == 0 or matrix == 2:
                    num_streams = 5000
                setting_message = f"Matrix visual: {matrix}"
            elif event.key == pygame.K_i:
                fade_intensity = min(5, fade_intensity + 0.1)
                setting_message = f"Fade intensity: {fade_intensity}"
            elif event.key == pygame.K_d:
                fade_intensity = max(0.01, fade_intensity - 0.1)
                setting_message = f"Fade intensity: {fade_intensity}"
            elif event.key == pygame.K_k:
                num_streams = min(100000, num_streams + 250)  # Increase the number of streams
                setting_message = f"Number of streams: {num_streams}"
            elif event.key == pygame.K_l:
                num_streams = max(10, num_streams - 250)  # Decrease the number of streams
                setting_message = f"Number of streams: {num_streams}"
            elif event.key == pygame.K_3:
                persistent_dot = not persistent_dot
                setting_message = f"Persistent dot: {'On' if persistent_dot else 'Off'}"
            elif event.key == pygame.K_4:
                # cycle font_color
                font_color, font_rgb = cycle_color(font_color)
                setting_message = f"Font color: {font_color}"
            elif event.key == pygame.K_5:
                # cycle font_decay_color
                font_decay_color, font_decay_rgb = cycle_color(font_decay_color)
                setting_message = f"Font decay color: {font_decay_color}"
            elif event.key == pygame.K_f:
                # Toggle fullscreen
                if screen.get_flags() & pygame.FULLSCREEN:
                    pygame.display.set_mode((width, height))
                else:
                    pygame.display.set_mode((width, height), pygame.FULLSCREEN)
                setting_message = f"Fullscreen: toggle"
            elif event.key == pygame.K_6:
                # Cycle background color
                background_color, _ = cycle_color(background_color)
                setting_message = f"Background color: {background_color}"
            elif event.key == pygame.K_7:
                color_cycle = not color_cycle
                setting_message = f"Color cycle: {color_cycle}"
            elif event.key == pygame.K_r:
                # Get a list of supported resolutions
                resolutions = pygame.display.list_modes()
                # Cycle through resolutions
                current_resolution = (current_resolution - 1) % len(resolutions)
                width, height = resolutions[current_resolution]
                height = int(height * 0.90)
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                setting_message = f"Resolution: {width}x{height}"
                new_stream = True
            elif event.key == pygame.K_s:
                save_settings()
            elif event.key == pygame.K_t:
                ocr_enabled = not ocr_enabled
                setting_message = f"OCR: {'On' if ocr_enabled else 'Off'}"
            elif event.key == pygame.K_c:
                contour_detection = not contour_detection
                setting_message = f"Contour detection: {'On' if contour_detection else 'Off'}"
            elif event.key == pygame.K_q:
                save_settings()
                sys.exit(0)
            elif event.key == pygame.K_8:
                edge_detection = not edge_detection
                setting_message = f"Edge detection: {'On' if edge_detection else 'Off'}"
            elif event.key == pygame.K_1:
                random_fade = not random_fade
                setting_message = f"Random fade: {random_fade} {fade_intensity}"
            elif event.key == pygame.K_0:
                font_color = default_font_color
                decay_color = default_decay_color
                background_color = default_background_color
                setting_message = f"Reset colors: {font_color} {decay_color} {background_color}"
            elif event.key == pygame.K_9:
                static_overlay = not static_overlay
                setting_message = f"Toggle static overlay: {static_overlay}"

    current_frame = screen.copy()

    if random_fade:
        fade_intensity = random.uniform(0.1, 5.0)
    
    if setting_message:
        print(setting_message)
        setting_message = False
    #    new_stream = True

    if new_stream:
        # Get the dimensions of the display
        info = pygame.display.Info()
        display_width = info.current_w
        display_height = info.current_h
        print(f"Screen resolution: {display_width} x {display_height}")

        streams = create_streams(matrix)
        new_stream = False

    if display_width != width or display_height != height:
        display_width = width
        display_height = height
        print(f"Screen resolution: {width} x {height}")

    # Clear screen
    screen.fill((0, 0, 0, 120))

    # Clear screen with a semi-transparent black rectangle
    fade_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    fade_surface.fill((0, 0, 0, edge_decay))
    fade_surface.fill((0, 0, 0, fade_intensity))
    screen.blit(fade_surface, (0, 0))

    if color_cycle:
        # cycle font_color
        font_color, font_rgb = cycle_color(font_color)

    # Update and draw streams
    new_streams = []
    if matrix == 0:
        for speed, x, y, color in streams:
            # Update color
            color[3] = max(0, color[3] - fade_intensity)  # Reduce alpha component

            # Draw character
            char = chr(random.randint(0x20, 0x7E))  # ASCII characters
            column = 0
            font.render_to(screen, (x, int(y)), char, color)

            # Update position
            y += speed
            if y > height:
                #y = 0
                color = list(colors[font_color])  # Reset color
                y = random.randint(-height, 0)
            new_streams.append((speed, x, y, color))
        streams = new_streams
    elif matrix == 1:
        for speed, x, y in streams:
            # Draw characters
            for i in range(int(y), height, font_size):
                intensity = min(1, max(0, (i - y) / height))
                color = [int(c * intensity) for c in colors[font_color]]
                decay_color = [int(c * (1 - intensity)) for c in colors[font_decay_color]]
                final_color = [c1 + c2 for c1, c2 in zip(color, decay_color)]
                char = chr(random.randint(0x20, 0x7E))  # ASCII characters
                font.render_to(screen, (x, i), char, final_color)
            
            # Update stream
            y += speed
            if y > height:
                y = random.randint(-height, 0)
            new_streams.append((speed, x, int(y)))
        streams = new_streams
    elif matrix == 2:
        for speed, x, y, color in streams:
            # Update color
            color[3] = max(0, color[3] - fade_intensity) # Reduce alpha component

            # Draw character
            char = chr(random.randint(0x20, 0x7E)) # ASCII characters

            # Calculate intensity
            intensity = min(1, max(0, y / height))

            # Calculate decay color
            decay_color = [int(c * (1 - intensity)) for c in colors[font_decay_color]]

            # Calculate trail color
            trail_color = [int(c * intensity) for c in colors[font_decay_color]]

            # Calculate final color
            final_color = [min(255, c1 + c2) for c1, c2 in zip(color, decay_color)]

            # Render character with final color
            font.render_to(screen, (x, int(y)), char, final_color)

            # Render trailing characters with trail color
            for i in range(int(y), height, font_size):
                font.render_to(screen, (x, random.randint(0, height)), char, trail_color)

            # Update position
            y += speed
            if y > height:
                #y = 0
                color = list(colors[font_color]) # Reset color
                y = random.randint(-height, 0)
            new_streams.append((speed, x, y, color))
        streams = new_streams

    elif matrix == 3:
        # Generate a random noise image
        static = np.random.randint(0, 255, (width, height, 3), dtype=np.uint8)

        # Convert the noise image to a Pygame surface
        static_surface = pygame.surfarray.make_surface(static)

        # Blit the static surface onto the screen
        screen.blit(static_surface, (0, 0))

    if static_overlay:
        # Generate a semi-transparent static surface
        static_surface = generate_tv_static(width, height, alpha=128)

        # Blit the static surface onto the screen
        screen.blit(static_surface, (0, 0))

    # OCR function
    if ocr_enabled:
        recog += 1
        if recog >= 100:
            pygame.image.save(screen, '/tmp/temp.png')
            text = pytesseract.image_to_string('/tmp/temp.png')
            if text:
                print(f"Recognized text: {text}")
            recog = 0

    # Draw the persistent dot
    if persistent_dot:
        pygame.draw.circle(screen, (0, 255, 0), (width//2, height//2), 3)

    # Display help menu
    if help_menu:
        help_surface = display_help_menu()
        screen.blit(help_surface, (0, 0))

    contour_frames = 10 
    if contour_detection:
        if frame_counter % contour_frames == 0:
            print(frame_counter, contour_frames)
            contour_surface = detect_and_draw_contours(screen)
            screen.blit(contour_surface, (0, 0))

    # Draw edges on the screen if edge detection is enabled
    if edge_detection:
        new_edge_surface = detect_edges(screen)
        edge_surface.blit(new_edge_surface, (0, 0))
        edge_surface.set_alpha(100)
        screen.blit(edge_surface, (0, 0))

    if frame_counter >= 1024:
        frame_counter = 0

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
