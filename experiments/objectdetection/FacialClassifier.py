#!/usr/bin/env python3
#
# realtime.py

import os
import cv2
from deepface import DeepFace
from transformers import pipeline
from yt_dlp import YoutubeDL  # Use yt-dlp for downloading YouTube videos
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class IntentDetector:
    def __init__(self, text_model: str = 'distilbert-base-uncased-finetuned-sst-2-english', image_backend: str = 'ssd', tokenizers_parallelism: str = 'false', frame_interval: int = 10):
        os.environ["TOKENIZERS_PARALLELISM"] = tokenizers_parallelism
        self.text_classifier = pipeline("sentiment-analysis", model=text_model)
        self.image_backend = image_backend
        self.frame_interval = frame_interval  # Set the frame processing interval

    def detect_intent_from_text(self, text: str) -> str:
        result = self.text_classifier(text)[0]
        intent = result['label'].lower()
        return intent

    def detect_intent_from_image(self, image_path: str) -> dict:
        try:
            analysis_results = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=True,
                detector_backend=self.image_backend,
                silent=True
            )
            return analysis_results
        except Exception as e:
            return {"error": str(e)}

    def _process_frame(self, frame):
        try:
            analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], detector_backend=self.image_backend, enforce_detection=False)
            return analysis
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def detect_intent_from_video(self, video_source: str):
        local_video = False
        if "youtube.com" in video_source or "youtu.be" in video_source:
            video_source = self._download_youtube_video(video_source)
            local_video = True  # Mark that we downloaded this video locally

        if video_source is None:
            print("Error: Could not download or access the video.")
            return

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process only every `self.frame_interval` frame
            if frame_count % self.frame_interval == 0:
                analysis = self._process_frame(frame)
                if analysis:
                    print(analysis)

            # Display the frame (optional)
            cv2.imshow('Video Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Delete the video only if it was downloaded locally
        if local_video and os.path.exists(video_source):
            try:
                os.remove(video_source)
                print(f"Deleted the video file: {video_source}")
            except Exception as e:
                print(f"Error deleting the video file: {str(e)}")

    def _download_youtube_video(self, url: str) -> str:
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': '/tmp/%(title)s.%(ext)s',  # Download to /tmp directory
            'merge_output_format': 'mp4',
            'quiet': True
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                video_title = info_dict.get('title', None)
                if video_title:
                    return f"/tmp/{video_title}.mp4"
                else:
                    print("Error: Could not retrieve video title.")
                    return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

# Example usage:
if __name__ == "__main__":
    intent_detector = IntentDetector(tokenizers_parallelism='false', frame_interval=100)

    text = "I am so happy today!"
    text_intent = intent_detector.detect_intent_from_text(text)
    print(f"Detected intent from text: {text_intent}")

    image_path = "/Users/kato/Pictures/face_test_image.jpg"
    image_analysis = intent_detector.detect_intent_from_image(image_path)
    print(f"Detected intent from image: {image_analysis}")

    video_path = "https://www.youtube.com/watch?v=n84hBgtzvxo"
    intent_detector.detect_intent_from_video(video_path)

    # Example with a local file (should not be deleted)
    # local_video_path = "/Users/kato/Videos/sample_video.mp4"
    # intent_detector.detect_intent_from_video(local_video_path)

