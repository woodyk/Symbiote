#!/usr/bin/env python3
#
# someone.py

import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load audio file
def load_audio_file(file_path):
    input_length = 16000
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

# Assume you have a directory of m4a files and corresponding labels
audio_dir = '/Users/kato/Desktop/audio/'
audio_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.m4a')]
labels = ... # You need to provide the labels for your audio files

# Load all audio files and convert them to data
X = np.array([load_audio_file(file) for file in audio_files])
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a sequential model
model = Sequential()

# Add a dense layer
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=2, activation='softmax')) # 2 is the number of classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
