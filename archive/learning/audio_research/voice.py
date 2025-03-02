#!/usr/bin/env python3
#
# voice.py

import scipy.signal as signal
import soundfile as sf

# Load the WAV file
audio_file = 'example.wav'
audio_data, sample_rate = sf.read(audio_file)

# Define the frequency range of interest
f_low = 5746
f_high = 5748

# Define the filter coefficients for a bandpass filter
b, a = signal.butter(4, [2*f_low/sample_rate, 2*f_high/sample_rate], btype='bandpass')

# Apply the filter to the audio data
filtered_audio = signal.filtfilt(b, a, audio_data)

# Write the filtered audio to a new WAV file
sf.write('filtered_audio.wav', filtered_audio, sample_rate)
