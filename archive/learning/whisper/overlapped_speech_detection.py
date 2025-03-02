#!/usr/bin/env python3
#
# overlapped_speech_detection.py

from pyannote.audio.tasks import OverlappedSpeechDetection
pipeline = OverlappedSpeechDetection(segmentation=model)
pipeline.instantiate(HYPER_PARAMETERS)
osd = pipeline("200_for.mp3")

print(osd)
