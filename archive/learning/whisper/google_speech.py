#!/usr/bin/env python3
#
# google_speech.py

from google.cloud import speech_v1p1beta1 as speech

def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()

    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        audio_channel_count=2,
        enable_separate_recognition_per_channel=True,
    )

    response = client.recognize(config=config, audio=audio)

    for i, result in enumerate(response.results):
        print("Transcript: {}".format(result.alternatives[0].transcript))

# Call the function
transcribe_audio("200_to.mp3")
