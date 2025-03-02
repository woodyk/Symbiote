#!/usr/bin/env python3
#
# youtubeAudio.py

import os
import re
import threading
from queue import Queue
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import youtube_dl
import pyttsx3

# Download configuration
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
}

# YouTube search & download functions
def get_youtube_video_info(search_term, max_results=10):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey="YOUR_API_KEY")

    search_response = youtube.search().list(
        q=search_term,
        type="video",
        part="id,snippet",
        order="viewCount",
        maxResults=max_results
    ).execute()

    return search_response.get("items", [])

def download_audio(video_id, ydl_opts):
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch:{video_id}", download=False)
        audio_url = info['entries'][0]['formats'][0]['url']
        ydl.download([audio_url])

# Playback functions
def play_audio(queue, engine, speed):
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)

    while not queue.empty():
        video_id = queue.get()
        mp3_file = f"{video_id}.mp3"

        if os.path.exists(mp3_file):
            engine.say(f"Playing {video_id}")
            engine.setProperty('rate', speed)
            engine.play(mp3_file)

            while engine.busy():
                time.sleep(1)

            os.remove(mp3_file)

# User interaction
print("Enter search term:")
search_term = input()
search_results = get_youtube_video_info(search_term, 10)

print("\nDownloading...")
queued_files = Queue()
threads = []

for result in search_results:
    video_id = result['id']['videoId']
    queued_files.put(video_id)

    download_thread = threading.Thread(target=download_audio, args=(video_id, ydl_opts))
    threads.append(download_thread)
    download_thread.start()

# Wait for download completion
for thread in threads:
    thread.join()

print("\nPlaying...")
engine = pyttsx3.init()
speed = 1.5
play_thread = threading.Thread(target=play_audio, args=(queued_files, engine, speed))
play_thread.start()

# Cleanup
while not queued_files.empty():
    time.sleep(1)

engine.stop()
