#!/usr/bin/env python3
#
# speech.py

import os
import re
import time
import subprocess
import tempfile
import threading
import pyaudio
import pyttsx3
import numpy as np
import sounddevice as sd
import simpleaudio as sa
import random
import speech_recognition as sr
from gtts import gTTS
from queue import Queue

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class SymSpeech():
    def __init__(self, monitor=False, settings=False):
        self.monitor = monitor 
        self.settings = settings 

        # Create a recognizer instance
        self.r = sr.Recognizer()

        # Keyword
        self.keywords = [ 
                    "symbiote",
                    "help",
                    "someone",
                    "bob"
                    ]

    def start_keyword_listen(self):
        q = Queue()
        t = threading.Thread(target=self.keyword_listen, args=(q,))
        t.start()
        t.join
        return q 

    def stop_keyword_listen(self):
        self.stop_listening.set()

    def keyword_listen(self, q=False):
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Start the stream in non-blocking mode
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=p.get_default_input_device_info()['index'], start=False)

        if self.settings['debug']: 
            print("Listening...")

        text = ''
        data = b''

        while True:
            try:
                # Start the stream
                stream.start_stream()

                # Read a chunk from the stream
                data += stream.read(8000, exception_on_overflow = False)

                if len(data) > 100000:
                    data = b''
                    continue

                # Use recognize_google to convert audio to text
                try:
                    # Convert the audio to text
                    text = self.r.recognize_google(sr.AudioData(data, sample_rate=16000, sample_width=2))
                except sr.UnknownValueError:
                    # Google Speech Recognition could not understand audio
                    if self.settings['debug']:
                        print("Google Speech Recognition could not understand audio")
                    continue
                except sr.RequestError as e:
                    # Could not request results from Google Speech Recognition service
                    if self.settings['debug']:
                        print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    continue
                except Exception as e:
                    if self.settings['debug']:
                        print(f"Unknown exception: {e}")
                    continue

            except IOError:
                pass

            if self.settings['debug']:
                if len(text) > 0:
                    print("Recognized: ", text)

            # Check if the keyword is in the recognized text
            for keyword in self.keywords:
                if keyword.lower() in text.lower():
                    data = b''
                    # Stop the stream
                    stream.stop_stream()

                    if self.settings['debug']:
                        print("Keyword detected!")

                    ready = "Yes?"
                    self.say(ready)
                    recorded = self.listen(5)

                    if recorded is None:
                        break

                    if self.monitor:
                        self.launch_window(recorded)
                    elif q:
                        q.put(recorded)
                    else:
                        return recorded

            text = '' 

            return text

    def text_to_morse(self, text):
        morse_code = ''
        MORSE_CODE_DICT = {'A':'.-', 'B':'-...',
                    'C':'-.-.', 'D':'-..', 'E':'.',
                    'F':'..-.', 'G':'--.', 'H':'....',
                    'I':'..', 'J':'.---', 'K':'-.-',
                    'L':'.-..', 'M':'--', 'N':'-.',
                    'O':'---', 'P':'.--.', 'Q':'--.-',
                    'R':'.-.', 'S':'...', 'T':'-',
                    'U':'..-', 'V':'...-', 'W':'.--',
                    'X':'-..-', 'Y':'-.--', 'Z':'--..',
                    '1':'.----', '2':'..---', '3':'...--',
                    '4':'....-', '5':'.....', '6':'-....',
                    '7':'--...', '8':'---..', '9':'----.',
                    '0':'-----', ', ':'--..--', '.':'.-.-.-',
                    '?':'..--..', '/':'-..-.', '-':'-....-',
                    '(':'-.--.', ')':'-.--.-'}

        for char in text:
            if char != ' ':
                # Converts the character to uppercase
                # as our dictionary has uppercase letters
                char = char.upper()
                # Checks if character is a valid Morse Code character
                if char in MORSE_CODE_DICT:
                    morse_char = MORSE_CODE_DICT[char]
                    morse_code += morse_char + ' '
                    for symbol in morse_char:
                        time.sleep(0.7)
                        if symbol == '.':
                            # Play a short tick for dot
                            self.play_random_ticks(0.1, 0.1, 0.1)
                        elif symbol == '-':
                            # Play a longer tick for dash
                            self.play_random_ticks(0.3, 0.3, 0.3)
                else:
                    morse_code += ' '
            else:
                # 1 space indicates different characters
                # 2 spaces indicates different words
                morse_code += ' '
                # Add a pause between words
                time.sleep(0.9)

        return morse_code

    def play_random_ticks(self, duration, min_interval, max_interval):
        # Generate a 1kHz sine wave, 1 second long
        sample_rate = 192000
        t = np.linspace(0, 1, sample_rate, False)
        tick_data = 0.5 * np.sin(2 * np.pi * 1000 * t)

        # Convert to 16-bit data
        tick_data = (tick_data * 32767).astype(np.int16)

        # Create a playable audio object
        tick = sa.WaveObject(tick_data, 1, 2, sample_rate)

        # Calculate the time to stop
        stop_time = time.time() + duration

        # Play ticks at random intervals until the stop time
        while time.time() < stop_time:
            # Play the tick sound
            tick.play()

            # Wait for a random interval
            time.sleep(random.uniform(min_interval, max_interval))

    '''
    def say(self, say_message, rate=1000, volume=1.0):
        threading.Thread(target=self.play_random_ticks, args=(30, 0.2, 0.6)).start()
        #self.text_to_morse(say_message)
        return

        # Initialize the speech engine
        engine = pyttsx3.init()

        # Set the rate and volume
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)

        # Reverse the content
        reversed_content = " ".join(say_message.split()[::-1])[::-1]

        # Split the reversed content into sentences
        sentences = re.split(r'(?<=[.!?]) +', reversed_content)

        # Say each sentence with a pause and change in volume (to simulate inflection)
        for i, sentence in enumerate(sentences):
            # Change the volume for every other sentence to simulate inflection
            #volume = 1.0 if i % 2 == 0 else 0.8
            #engine.setProperty('volume', volume)

            # Say the sentence and then pause
            #engine.say(sentence)
            #engine.runAndWait()

            # Calculate the duration for the ticks based on the number of words in the sentence
            words = len(sentence.split())
            duration = words / 200 * 60  # 200 words per minute

            # Play the ticks
            self.play_random_ticks(30, 0.0, 0.1)

    '''
    def say(self, say_message, speed=1.0):
        try:
            text_to_speech = gTTS(text=say_message)
        except Exception as e:
            print(f"Unable to get text to speech. {e}")
            return

        # save the speech audio into a file
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            tempfile_path = f"{fp.name}.mp3"

        text_to_speech.save(tempfile_path)

        # play the audio file
        pygame.mixer.init()
        pygame.mixer.music.load(tempfile_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy() == True:
            continue

        os.remove(tempfile_path)

    def listen(self, duration):
        request_text = str()
        with sr.Microphone() as source:
            if self.settings['debug']:
                print("Speak:")

            # read the audio data from the default microphone
            audio_data = self.r.record(source, duration=5)
            if self.settings['debug']:
                print("Recognizing...")

            # convert speech to text
            try:
                request_text = self.r.recognize_google(audio_data)
            except:
                # Google Speech Recognition could not understand audio
                requested_text = None 

            if self.settings['debug']:
                print(request_text)

            if len(request_text) < 2:
                request_text = None

            return request_text


    def launch_window(self, content):
        command = ["terminator", "-e"]
        issue_command = f'symbiote -q "{content}" -e'
        command.append(issue_command)

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        while process.poll() is None:
            time.sleep(1)

        if self.settings['debug']:
            print("Terminal closed.")

        return
