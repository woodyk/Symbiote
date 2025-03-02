#!/usr/bin/env python3
#
# symbios.py

import asyncio
import websockets
import pygame

# Initialize Pygame
pygame.init()

# Set up some constants for the Pygame window
WIDTH, HEIGHT = 640, 480
DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up a set to hold the connected WebSocket clients
clients = set()

async def server(websocket, path):
    # Add the WebSocket client to the set of clients
    clients.add(websocket)
    try:
        # Main loop: wait for a message, then broadcast it to all clients
        async for message in websocket:
            await broadcast(message)
    finally:
        # If a client disconnects, remove it from the set of clients
        clients.remove(websocket)

async def broadcast(message):
    # Send the message to all connected clients
    if clients:
        await asyncio.wait([client.send(message) for client in clients])

start_server = websockets.serve(server, "localhost", 8765)

# Start the WebSocket server
asyncio.get_event_loop().run_until_complete(start_server)

# Main Pygame loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Update the Pygame display
    pygame.display.flip()

# Run the WebSocket server forever
asyncio.get_event_loop().run_forever()

