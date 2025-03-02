#!/usr/bin/env python3
#
# vision300.py

import pygame
import numpy as np
import cv2
import pytesseract

# Define the size of the image
width, height = 1024, 768 
fps = 20

# Ratio of white to black pixels (0.1 means 10% white, 90% black)
ratio = 0.01

# Static Mode
static_mode = False
min_blob_size = 10 
decay_rate = .01 
blob_radius = 5
blob_detection = True
connect_blobs = False
text_detect = False
static_alpha = 75  # initial alpha value for static
static_on = True

# Matrix Mode
matrix_mode = False
matrix_density = 0.1
matrix_persistence = 0.5
matrix_speed = 1.0
matrix_color = (255, 255, 255) # White
matrix_decay_color = (0, 255, 0) # Green

# Other global variables
old_blobs = []
running = True
paused = False
help_menu = False
text_buffer = ''
frame_count = 0
detected_text = ''

# Initialize Pygame
pygame.init()

# Set up some necessities
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36) # You can change None to a font file path to use a specific font

def blob_detection(binary_data, min_size=256):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_data, connectivity=8)

    blobs = []
    for i in range(1, num_labels):
        # Ignore small blobs
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            continue

        # Add blob to list
        blob = {
            "label": i,
            "size": stats[i, cv2.CC_STAT_AREA],
            "centroid": (centroids[i, 0], centroids[i, 1]),
            "bbox": (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            ),
        }
        blobs.append(blob)

    return blobs, labels

def decay_blobs(old_blobs, new_blobs, decay_rate=.5):
    # Create a list to hold the updated blobs
    updated_blobs = []

    # For each old blob
    for old_blob in old_blobs:
        # Check if it is in the new blobs
        match = next((blob for blob in new_blobs if blob["label"] == old_blob["label"]), None)

        # If the blob is in the new blobs, keep it and don't decay
        if match:
            updated_blobs.append(old_blob)
            new_blobs.remove(match) # Remove the matched blob from the new blobs
        else:
            # If the blob is not in the new blobs, decay it
            old_blob["size"] *= (1 - decay_rate)

            # Only keep the blob if it's size is still above a certain threshold (e.g., 1)
            if old_blob["size"] > 1:
                updated_blobs.append(old_blob)

    # Add any remaining new blobs to the updated blobs list
    updated_blobs.extend(new_blobs)

    return updated_blobs

def draw_blobs(blobs, labels, width, height):
    # Create a new surface with an alpha channel
    surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # Draw each blob onto the surface
    for blob in blobs:
        # Determine the color and size of the blob
        intensity = int(blob["size"])  # Use blob size to determine green intensity
        intensity = min(255, max(0, intensity))  # Clamp intensity between 0 and 255
        intensity = min(255, max(0, intensity))  # Clamp intensity between 0 and 255
        color = (255, 255, 255, 255)  # Green color, with intensity based on blob size and alpha set to maximum

        # Get the pixels for this blob
        blob_pixels = np.where(labels == blob["label"])

        # Draw the blob
        for i in range(len(blob_pixels[0])):
            pygame.draw.circle(surface, color, (blob_pixels[1][i], blob_pixels[0][i]), 1)

    if connect_blobs:
        # Draw lines to the two closest blobs for each blob
        for blob in blobs:
            # Compute the distances to all other blobs
            distances = [((other_blob["centroid"][0] - blob["centroid"][0]) ** 2
                          + (other_blob["centroid"][1] - blob["centroid"][1]) ** 2, other_blob)
                         for other_blob in blobs if other_blob != blob]
            # Find the two closest blobs
            closest_blobs = sorted(distances, key=lambda x: x[0])[:2]

            # Draw lines to the two closest blobs
            for _, close_blob in closest_blobs:
                start_pos = (int(blob["centroid"][0]), int(blob["centroid"][1]))
                end_pos = (int(close_blob["centroid"][0]), int(close_blob["centroid"][1]))
                pygame.draw.line(surface, (0, 255, 0), start_pos, end_pos, 1)

    return surface

def detect_text(img):
    # Convert the Pygame surface (RGB format) to OpenCV format (BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to detect text
    text = pytesseract.image_to_string(gray)

    if len(text) > 0:
        print(text, end="", flush=True)

    return text

# Create an array of random numbers using the specified ratio
def create_static(height, width, ratio):
    # Create an array of random floats from a uniform distribution
    random_floats = np.random.uniform(0, 1, (height, width))

    # Create a binary mask where values less than the ratio are True (these will become our white pixels)
    mask = random_floats < ratio

    # Use the mask to create an array of black and white pixels
    static_data = np.where(mask, 255, 0).astype(np.uint8)

    return static_data

def create_surface_with_alpha(data, alpha):
    # Create an RGB surface from the data
    rgb_surface = pygame.surfarray.make_surface(data)

    # Create a new surface with the same size as the RGB surface, and SRCALPHA to allow alpha blending
    rgba_surface = pygame.Surface(rgb_surface.get_size(), pygame.SRCALPHA)

    # Fill the RGBA surface with the RGB surface and the alpha value
    rgba_surface.fill((0, 0, 0, alpha))
    rgba_surface.blit(rgb_surface, (0, 0))

    return rgba_surface

def display_help_menu():
    help_text = [
        "Help Menu:",
        "Up Arrow: Increase ratio of white pixels",
        "Down Arrow: Decrease ratio of white pixels",
        "H: Show this help menu",
        "S: Toggle Static mode",
        "Space: Pause/Resume",
        "B: Increase minimum blob size",
        "N: Decrease minimum blob size",
        "D: Increase decay rate",
        "C: Decrease decay rate",
        "R: Increase blob radius",
        "E: Decrease blob radius",
        "L: Toggle blob connections",
        "1: Toggle static background",
        "2: Toggle text detection",
        "3: Toggle blob detection",
    ]

    help_text.extend([
         "M: Toggle Matrix mode",
         "U: Increase Matrix density",
         "J: Decrease Matrix density",
         "I: Increase Matrix persistence",
         "K: Decrease Matrix persistence",
         "O: Increase Matrix speed",
         "L: Decrease Matrix speed",
         "P: Change Matrix color",
         "Q: Change Matrix decay color",
        ])

    for i, line in enumerate(help_text):
        text_surface = font.render(line, True, (255, 255, 255)) # White text
        screen.blit(text_surface, (10, 10 + 40*i)) # 40 pixels line height

def create_matrix(height, width, density, persistence, speed, color, decay_color):
    # Create an empty array for the Matrix data
    matrix_data = np.zeros((height, width, 3), dtype=np.uint8)

    # Create an array of random floats for the character positions
    positions = np.random.uniform(0, 1, (height, width))

    # Create a mask where values less than the density are True (these will become our characters)
    mask = positions < density

    # Use the mask to set the color of the characters
    matrix_data[mask] = color

    # Create an array of random floats for the decay
    decay = np.random.uniform(0, 1, (height, width))

    # Create a mask where values less than the persistence are True (these will become our decayed characters)
    mask = decay < persistence

    # Use the mask to set the color of the decayed characters
    matrix_data[mask] = decay_color

    # Shift the rows of the matrix data down by the speed
    matrix_data = np.roll(matrix_data, int(speed), axis=0)

    return matrix_data



coverground = pygame.Surface((width, height))
coverground.fill((0, 0, 0))

# Create a list to hold the dirty rectangles
dirty_rects = []

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                ratio = min(1.0, ratio + 0.001)
            elif event.key == pygame.K_DOWN:
                ratio = max(0.0, ratio - 0.001)
            elif event.key == pygame.K_h:
                help_menu = not help_menu # Toggle help menu
            elif event.key == pygame.K_SPACE:
                paused = not paused # Toggle pause
            elif event.key == pygame.K_b:
                min_blob_size += 10
            elif event.key == pygame.K_n:
                min_blob_size = max(10, min_blob_size - 10)
            elif event.key == pygame.K_d:
                decay_rate = min(1.0, decay_rate + 0.01)
            elif event.key == pygame.K_c:
                decay_rate = max(0.0, decay_rate - 0.01)
            elif event.key == pygame.K_r:
                blob_radius += 1
            elif event.key == pygame.K_e:
                blob_radius = max(1, blob_radius - 1)
            elif event.key == pygame.K_l:
                connect_blobs = not connect_blobs # Toggle blob connections
            elif event.key == pygame.K_1:
                static_on = not static_on
            elif event.key == pygame.K_2:
                text_detect = not text_detect
            elif event.key == pygame.K_3:
                blob_detection = not blob_detection
            elif event.key == pygame.K_m:
                matrix_mode = not matrix_mode # Toggle Matrix mode
            elif event.key == pygame.K_u:
                matrix_density = min(1.0, matrix_density + 0.01)
            elif event.key == pygame.K_j:
                matrix_density = max(0.0, matrix_density - 0.01)
            elif event.key == pygame.K_i:
                matrix_persistence = min(1.0, matrix_persistence + 0.01)
            elif event.key == pygame.K_k:
                matrix_persistence = max(0.0, matrix_persistence - 0.01)
            elif event.key == pygame.K_o:
                matrix_speed += 1
            elif event.key == pygame.K_l:
                matrix_speed = max(1, matrix_speed - 1)
            elif event.key == pygame.K_p:
                matrix_color = (255, 255, 255) # Change to a different color
            elif event.key == pygame.K_q:
                matrix_decay_color = (0, 255, 0) # Change to a different color
            elif event.key == pygame.K_s:
                static_mode = not static_mode

    if not paused:
        frame_count += 1

        if static_mode:
            # Generate static data
            static_data = create_static(height, width, ratio)

            # Create static surface
            static_surface = pygame.surfarray.make_surface(static_data)
            static_surface = pygame.transform.scale(static_surface, (width, height))

            # Blit the static surface onto the screen
            if static_on:
                screen.blit(static_surface, (0, 0))
            else:
                screen.blit(coverground, (0, 0))

            # Threshold static data to binary
            _, binary_data = cv2.threshold(static_data, 127, 255, cv2.THRESH_BINARY)

            if blob_detection:
                # Detect blobs in binary data
                blobs, labels = blob_detection(binary_data, min_blob_size)

                # Compare old blobs to new blobs and decay if not persisting
                old_blobs = decay_blobs(old_blobs, blobs, decay_rate)

                if blobs:
                    # Draw blobs
                    blob_surface = draw_blobs(old_blobs, labels, width, height)
                    rect = pygame.Rect(0, 0, width, height)
                    screen.blit(blob_surface, rect)
                    dirty_rects.append(rect)
                    #screen.blit(blob_surface, (0, 0)) 

                # Only update the dirty rectangles
                pygame.display.update(dirty_rects)

                # Clear the list of dirty rectangles for the next frame
                dirty_rects = []

        # Convert the current frame to a NumPy array
        if text_detect:
            img = pygame.surfarray.array3d(screen)
            detect_text(img)

        if matrix_mode:
             # Generate Matrix data
             matrix_data = create_matrix(height, width, matrix_density, matrix_persistence, matrix_speed, matrix_color, matrix_decay_color)

             # Create Matrix surface
             matrix_surface = pygame.surfarray.make_surface(matrix_data)
             matrix_surface = pygame.transform.scale(matrix_surface, (width, height))

             # Blit the Matrix surface onto the screen
             screen.blit(matrix_surface, (0, 0))

    if help_menu:
        display_help_menu()

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
