#!/usr/bin/env python3
#
# seismon.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import datetime

# Set the serial port and baud rate
serialPort = "/dev/cu.usbserial-210"
baudRate = 115200

# Open the serial port
ser = serial.Serial(serialPort, baudRate)

# Create a figure for plotting
fig, axs = plt.subplots(2)
fig.suptitle('Live Graph from Serial Port')

# Create lists for each sensor data
ax_data = []
ay_data = []
az_data = []
gx_data = []
gy_data = []
gz_data = []
time_data = []

# This function is called periodically from FuncAnimation
def animate(i, axs):

    # Read data from serial port
    line = ser.readline().strip().decode()  # assuming the data is a string of CSV values

    # Skip if line is empty
    if not line:
        return

    # Split the line into strings, skip any empty strings, and convert the rest to floats
    try:
        data = [float(x) for x in line.split(',') if x]
    except Exception as e:
        return

    # Limit the number of data points to 50
    if len(ax_data) > 50:
        ax_data.pop(0)
        ay_data.pop(0)
        az_data.pop(0)
        gx_data.pop(0)
        gy_data.pop(0)
        gz_data.pop(0)

    # Get current time
    now = datetime.datetime.now().strftime('%H:%M:%S')

    # Add data to lists
    ax_data.append(data[0])
    ay_data.append(data[1])
    az_data.append(data[2])
    gx_data.append(data[3])
    gy_data.append(data[4])
    gz_data.append(data[5])
    time_data.append(now)

    # Draw x and y lists
    axs[0].clear()
    axs[0].plot(ax_data, label='ax')
    axs[0].plot(ay_data, label='ay')
    axs[0].plot(az_data, label='az')
    axs[0].legend(loc='upper left')
    axs[0].set_ylabel('Acceleration')

    axs[1].clear()
    axs[1].plot(gx_data, label='gx')
    axs[1].plot(gy_data, label='gy')
    axs[1].plot(gz_data, label='gz')
    axs[1].legend(loc='upper left')
    axs[1].set_ylabel('Gyro')

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(axs,), interval=50)
plt.show()

