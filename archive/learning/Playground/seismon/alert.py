#!/usr/bin/env python3
#
# seismon.py

#!/usr/bin/env python3
#
# tt.py

import re
import serial
import queue
import threading
import time
import sys
import pandas as pd
from io import StringIO
import pygame
import random
import modules.graph as graph

pygame.init()

# Set up some constants
WIDTH, HEIGHT = 640, 480
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
MAX_POINTS = 1000

# Create the display surface
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Create a Graph object
graph = graph.Graph(screen, WIDTH, HEIGHT, MAX_POINTS, BLUE)


# Set the serial port and baud rate
serialPort = "/dev/cu.usbserial-210"
baudRate = 115200

# Create a queue to hold the data
queue = queue.Queue(maxsize=10000) # Increase the size of the queue
columns = []

debugs = False

def serialInit():
    print("Waiting for serial to init...")
    # Try to open the serial port
    while True:
        try:
            ser = serial.Serial(serialPort, baudRate)
            print("Device initialized...")
            return ser
        except serial.SerialException as e:
            debug(f"device: {e}")

        time.sleep(1) # Wait for 1 second before trying again

def getColumns(ser):
    print("Getting column names...")
    columns = []
    line = str()
    while True:
        try:
            line = ser.readline().strip().decode()
        except Exception as e:
            debug(f"error: {e}")

        if line:
            match = re.search(r'columns', line)
            if match:
                debug(repr(line))
                col_names = line.split(":")
                cline = col_names[1]
                columns = cline.split(",")
                return columns

        time.sleep(1)

def debug(*args, **kwargs):
    if debugs:
        print(*args, file=sys.stderr, **kwargs)

def read_serial(queue, ser):
    # Start reading data
    print("Begin reading serial...")
    while True:
        try:
            line = ser.readline().strip().decode()
        except Exception as e:
            debug(f"error: {e}")
            continue

        if line:
            try:
                # Create a pandas DataFrame from the CSV line
                df = pd.read_csv(StringIO(line), header=None)

                # Convert the DataFrame to a list and add it to the queue
                data = df.values.tolist()
                queue.put(data)
                debug(data)
            except Exception as e:
                debug(f"error: {e} {line}")

def flush_queue(queue):
    debug("flushing queue")
    while True:
        try:
            queue.get_nowait()
        except:
            pass

def process_data(queue):
    data = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        graph.add_data(random.randint(0, HEIGHT))

        screen.fill(WHITE)
        graph.draw_line()
        pygame.display.flip()

        data.append(queue.get())
        if len(data) >= 300:
            data.clear()
     

def map_value(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

ser = serialInit()
threading.Thread(target=read_serial, args=(queue, ser)).start()
process_data(queue);
pytame.quit()
