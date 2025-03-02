#!/usr/bin/env python3
#
# dubstep.py

from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
from random import randint

# Define the tempos (in beats per minute)
tempos = [120, 140, 160]

# Define the beat rates
beat_rates = [.25, .5, .75, 1, 2, 4, 6, 8, 16]

# Define the frequencies for the dubstep sounds
frequencies = [randint(20, 100) for _ in range(10)]

# Create an empty audio segment
song = AudioSegment.empty()

# Generate the dubstep sounds
for i in range(100):
    # Choose a random tempo
    tempo = tempos[randint(0, len(tempos) - 1)]

    # Choose a random beat rate
    beat_rate = beat_rates[randint(0, len(beat_rates) - 1)]

    # Define the beat (in milliseconds)
    beat = 3000 / tempo * beat_rate

    # Choose a random frequency
    freq = frequencies[randint(0, len(frequencies) - 1)]

    # Generate a sine wave with the chosen frequency
    sine_wave = Sine(freq)

    # Generate a sound with the length of a beat
    sound = sine_wave.to_audio_segment(duration=beat)

    # Add the sound to the song
    song += sound

# Play the song
play(song)

