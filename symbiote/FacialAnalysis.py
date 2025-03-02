#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: FacialAnalysis.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-27 06:24:45
# FacialAnalysis.py

import os
import cv2
import json
import requests
from deepface import DeepFace
from yt_dlp import YoutubeDL
from PIL import Image, ImageDraw, ImageFont
import tempfile
import warnings
import logging
import tensorflow as tf
from tqdm import auto as tqdm
import sys
import contextlib

def suppress_logs():
    """Suppress unnecessary logs from TensorFlow, Keras, and other libraries."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    logging.getLogger('deepface').setLevel(logging.ERROR)
    tqdm.tqdm = tqdm.tqdm(disable=True)
    warnings.simplefilter(action='ignore', category=FutureWarning)


class FacialAnalyzer:
    def __init__(self, image_backend: str = 'ssd', tokenizers_parallelism: str = 'false'):
        os.environ["TOKENIZERS_PARALLELISM"] = tokenizers_parallelism
        self.image_backend = image_backend

    def load_image(self, image_source):
        if image_source.startswith(('http://', 'https://')):
            return Image.open(requests.get(image_source, stream=True).raw)
        return Image.open(image_source)

    def analyze_image(self, image_source: str, verbose: bool = False):
        image = self.load_image(image_source)

        with self.suppress_stdout():
            analysis_results = self._analyze_faces(image_source, verbose)

        if analysis_results is None:
            return None, None

        labeled_image = self._draw_boxes(image, analysis_results)
        if verbose:
            print(json.dumps(analysis_results, indent=2))
        return labeled_image, analysis_results

    def _analyze_faces(self, image_source: str, verbose: bool = False):
        try:
            return DeepFace.analyze(
                img_path=image_source,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=True,
                detector_backend=self.image_backend,
                silent=not verbose
            )
        except Exception as e:
            if verbose:
                print(f"Error analyzing faces: {e}")
            return None

    def _draw_boxes(self, image, analysis_results):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for i, result in enumerate(analysis_results):
            region = result['region']
            box = (region['x'], region['y'], region['x'] + region['w'], region['y'] + region['h'])
            draw.rectangle(box, outline="red", width=3)

            label_text = f"Person {i+1}\nGender: {result['dominant_gender']}, Age: {result['age']}"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
            text_location = (box[0], box[1] - text_size[1] - 10 if box[1] - text_size[1] - 10 > 0 else box[1])

            background_box = [text_location[0] - 5, text_location[1] - 5,
                              text_location[0] + text_size[0] + 5, text_location[1] + text_size[1] + 5]
            draw.rectangle(background_box, fill=(0, 0, 0, 128))

            draw.text(text_location, label_text, fill="white", font=font)

        return image

    def display_image(self, image):
        image.show()

    def analyze_video(self, video_source: str, frame_interval: int = 10, show_video: bool = False, verbose: bool = False):
        video_source, local_video = self._handle_video_source(video_source)
        if video_source is None:
            if verbose:
                print("Error: Could not download or access the video.")
            return None

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            if verbose:
                print("Error: Could not open video.")
            return None

        results = self._process_video_frames(cap, frame_interval, show_video, verbose)

        cap.release()
        if show_video:
            cv2.destroyAllWindows()

        if local_video and os.path.exists(video_source):
            self._cleanup_video_file(video_source, verbose)

        return results

    def _handle_video_source(self, video_source: str):
        if "youtube.com" in video_source or "youtu.be" in video_source:
            return self._download_youtube_video(video_source), True
        elif video_source.startswith(('http://', 'https://')):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            self._download_video_from_url(video_source, temp_file.name)
            return temp_file.name, True
        return video_source, False  # Assuming it's a local file path

    def _process_video_frames(self, cap, frame_interval: int, show_video: bool, verbose: bool):
        frame_count = 0
        results = []
        with self.suppress_stdout():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    analysis = self._process_frame(frame, verbose)
                    if analysis:
                        results.append({'frame': frame_count, 'analysis': analysis})

                if show_video:
                    cv2.imshow('Video Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
        return results

    def _process_frame(self, frame, verbose: bool):
        try:
            analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'],
                                        detector_backend=self.image_backend, enforce_detection=False, silent=not verbose)
            if analysis and isinstance(analysis, list) and len(analysis) > 0:
                result = {
                    'age': analysis[0].get('age', None),
                    'gender': analysis[0].get('gender', None),
                    'dominant_emotion': analysis[0].get('dominant_emotion', None),
                    'dominant_race': analysis[0].get('dominant_race', None)
                }
                if verbose:
                    print(f"Frame analysis: {result}")
                return result
        except Exception as e:
            if verbose:
                print(f"Error processing frame: {e}")
        return None

    def _cleanup_video_file(self, video_source: str, verbose: bool):
        try:
            os.remove(video_source)
            if verbose:
                print(f"Deleted the video file: {video_source}")
        except Exception as e:
            if verbose:
                print(f"Error deleting the video file: {e}")

    def _download_youtube_video(self, url: str) -> str:
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': '/tmp/%(title)s.%(ext)s',
            'merge_output_format': 'mp4',
            'quiet': True
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                return f"/tmp/{info_dict.get('title', None)}.mp4"
        except Exception as e:
            return None

    def _download_video_from_url(self, url: str, output_path: str):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                print(f"Failed to download video. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while downloading the video: {e}")

    @contextlib.contextmanager
    def suppress_stdout(self):
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


# Example usage:
if __name__ == "__main__":
    suppress_logs()  # Apply log suppression

    analyzer = FacialAnalyzer(tokenizers_parallelism='false', image_backend="mtcnn")
    image_source = "https://www.jazzastudios.com/cdn/shop/products/FaceRefCover1.jpg?v=1659943039&width=1920"
    labeled_image, analysis_results = analyzer.analyze_image(image_source, verbose=False)
    print(json.dumps(analysis_results, indent=4))

    if labeled_image:
        analyzer.display_image(labeled_image)

    analyzer = FacialAnalyzer(tokenizers_parallelism='false', image_backend="ssd")
    video_path = "https://www.youtube.com/watch?v=n84hBgtzvxo"
    video_analysis = analyzer.analyze_video(video_path, frame_interval=100, show_video=False, verbose=False)
    print(json.dumps(video_analysis, indent=4))

