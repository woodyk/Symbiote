#!/usr/bin/env python3
#
# create_model.py

import tensorflow as tf
from tensorflow.keras import layers
import serial
import numpy as np

def create_model():
    # Create a Sequential model
    model = tf.keras.Sequential()

    # Add the input layer
    model.add(layers.Dense(64, activation='relu', input_shape=(100,)))

    # Add hidden layers
    # The exact structure of these layers might need to be adjusted depending on your specific requirements
    model.add(layers.Dense(64, activation='relu'))

    # Add the output layer
    model.add(layers.Dense(10))

    return model

def train_model(model, com_port, baud_rate, num_samples):
    # Open the COM port
    ser = serial.Serial(com_port, baud_rate)

    # Initialize an empty list to hold the training data
    data = []

    # Read data from the COM port
    for _ in range(num_samples):
        line = ser.readline()
        # Convert the line from bytes to string and strip trailing newlines
        line = line.decode('utf-8').strip()
        # Convert the line to a list of floats
        sample = list(map(float, line.split(',')))
        # Add the sample to the data
        data.append(sample)

    # Convert the data to a numpy array
    data = np.array(data)

    # Split the data into features and labels
    # This assumes that the last column of the data is the label
    features = data[:, :-1]
    labels = data[:, -1]

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    model.fit(features, labels, epochs=10)

    # Close the COM port
    ser.close()

# Test the function
model = create_model()
train_model(model, 'COM3', 9600, 1000)

