#!/usr/bin/env python3
#
# ping_map.py

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Constants
fs = 44100  # Sample rate 
seconds = 3  # Duration of recording

# Generate a 1kHz sine wave
f = 10000  # Frequency in Hz
t = np.arange(fs) / fs  # Time array
signal = 0.5 * np.sin(2 * np.pi * f * t)  # 1kHz sine wave

# Play the sound
sd.play(signal, fs)

# Record audio for a few seconds
print("Recording...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
print("Recording finished")

# Write recording to a WAV file
write('output.wav', fs, recording)

# Load the original sound and the recorded sound
original_sound = signal
recorded_sound = recording[:, 0]  # Use the first channel of the recording

# Perform a cross-correlation between the original sound and the recorded sound
corr = correlate(recorded_sound, original_sound)

# Find the index of the maximum value of the cross-correlation
delay_index = np.argmax(corr)

# Estimate the time delay of the echo
delay_seconds = delay_index / fs

print(f"Estimated time delay of the echo: {delay_seconds} seconds")

# Plot the cross-correlation
plt.plot(corr)
plt.title('Cross-correlation between the original sound and the recorded sound')
plt.xlabel('Index')
plt.ylabel('Cross-correlation')
plt.show()

