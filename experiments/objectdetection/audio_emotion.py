#!/usr/bin/env python3
#
# audio_emotion.py

import os
import numpy as np
import torch
import yt_dlp as youtube_dl
import wave
import pyaudio
from pydub import AudioSegment
from speechbrain.inference.interfaces import foreign_class

# Step 1: Download the video and extract audio
def download_audio(youtube_url, output_file="tmp.wav"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_file,  # Save as tmp.wav
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Handle potential file renaming by yt-dlp
    if not os.path.exists(output_file):
        if os.path.exists(f"{output_file}.wav"):
            output_file = f"{output_file}.wav"

    return output_file

# Step 2: Split the audio into chunks
def split_audio_into_chunks(wav_file, chunk_duration):
    audio = AudioSegment.from_wav(wav_file)
    chunk_length_ms = chunk_duration * 1000  # Convert to milliseconds
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

# Step 3: Play audio and process each chunk for emotion detection
def process_chunks(chunks, playback_speed=1.0):
    # Load the pre-trained emotion recognition model using foreign_class
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )

    p = pyaudio.PyAudio()

    for i, chunk in enumerate(chunks):
        # Adjust playback speed
        playback_chunk = chunk._spawn(chunk.raw_data, overrides={"frame_rate": int(chunk.frame_rate * playback_speed)})
        playback_chunk = playback_chunk.set_frame_rate(chunk.frame_rate)

        # Play the audio chunk
        stream = p.open(format=p.get_format_from_width(playback_chunk.sample_width),
                        channels=playback_chunk.channels,
                        rate=playback_chunk.frame_rate,
                        output=True)
        stream.write(playback_chunk.raw_data)
        stream.stop_stream()
        stream.close()

        # Convert chunk to numpy array
        samples = np.array(playback_chunk.get_array_of_samples())
        audio_np = samples.astype(np.float32) / 32768.0  # Normalize the data
        audio_tensor = torch.tensor([audio_np])

        # Use the classifier for emotion detection
        out_prob, score, index, text_lab = classifier.classify_batch(audio_tensor)
        print(out_prob, score, index, text_lab)
        print(f"Chunk {i + 1}: Detected Emotion: {text_lab}, Confidence: {score.item()}")

    p.terminate()

# Main function
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=-BIDA_6t3VA"
    wav_file = download_audio(youtube_url, "tmp")
    chunks = split_audio_into_chunks(wav_file, chunk_duration=5)  # Set chunk_duration to 5 seconds
    process_chunks(chunks, playback_speed=2.0)  # Play at normal speed
    os.remove(wav_file)

