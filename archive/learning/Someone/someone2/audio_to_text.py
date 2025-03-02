#!/usr/bin/env python3
#
# audio_to_text.py

import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Audio file as source
# listening the audio file and store in audio_text variable

with sr.AudioFile('/Users/kato/Desktop/audio/1.wav') as source:
    
    audio_text = r.listen(source)
    
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    
    try:
        
        # using google speech recognition
        print("Converting audio transcripts into text ...")
        text = r.recognize_sphinx(audio_text)
        print(text)
     
    except:
         print('Sorry.. run again...')

