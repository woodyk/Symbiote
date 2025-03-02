#!/usr/bin/env python3
#
# audio_analyze.py

import tensorflow as tf
import soundfile as sf
import numpy as np

# Load the WAV file
audio_file = 'example.wav'
audio_data, sample_rate = sf.read(audio_file)

# Normalize the audio data
audio_data = audio_data / np.max(np.abs(audio_data))

print(audio_data.shape)


# Define the labels
labels = np.zeros((len(audio_data), 2))
labels[:len(audio_data)//2, 0] = 1
labels[len(audio_data)//2:, 1] = 1

# Define the model architecture
model = tf.keras.Sequential([
 tf.keras.layers.Dense(64, activation='relu', input_shape=(len(audio_data),)),
 tf.keras.layers.Dense(64, activation='relu'),
 tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape the audio data to have shape (1, num_samples)
audio_data = np.expand_dims(audio_data, axis=0)

model.fit(audio_data, labels, epochs=10, batch_size=32)
model.save("audio_model.h5")
