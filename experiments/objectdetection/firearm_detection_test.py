#!/usr/bin/env python3
#
# firearm_detection_test.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
from yt_dlp import YoutubeDL
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import json

MODEL = "openai/clip-vit-base-patch32"
MODEL = "openai/clip-vit-large-patch14"

# Load the CLIP model and processor
model = CLIPModel.from_pretrained(MODEL)
processor = CLIPProcessor.from_pretrained(MODEL)

# Define the text prompts for detecting firearms
firearm_classes = ["gun", "pistol", "rifle",
                   "shotgun", "general"]

def get_video_source(source='webcam', youtube_url=None):
    if source == 'youtube' and youtube_url:
        ydl_opts = {
            'format': 'best',
            'quiet': True
        }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_url = info_dict.get("url", None)
        return cv2.VideoCapture(video_url)
    else:
        # Default to webcam
        return cv2.VideoCapture(0)

def get_timestamp(frame_number, fps):
    """Calculate timestamp in seconds from frame number and frame rate."""
    return frame_number / fps

def detect_firearm_in_frame(frame):
    """Detect if a firearm is present in the frame using CLIP."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=firearm_classes, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # Label probabilities

    # Find the most likely class
    max_prob, max_index = torch.max(probs, dim=1)
    return max_prob.item(), firearm_classes[max_index.item()]

def main(source='webcam', youtube_url=None, frame_skip=1):
    # Initialize the video source
    cap = get_video_source(source, youtube_url)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Process only every `frame_skip`th frame
        if frame_number % frame_skip != 0:
            continue

        # Detect firearm in the current frame
        prob, detected_class = detect_firearm_in_frame(frame)

        # If a firearm is detected with a high probability, log and draw bounding box
        if prob > 0.5:  # You can adjust this threshold based on your needs
            timestamp = get_timestamp(frame_number, fps)
            detection_info = {
                "event": "firearm_detected",
                "object_name": detected_class,
                "confidence": round(float(prob), 2),
                "timestamp": timestamp
            }

            # Print the detection info as a JSON string
            print(json.dumps(detection_info, indent=4))

            # Draw a label on the frame
            label = f'{detected_class} {prob:.2f}'
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with labels
        cv2.imshow('CLIP Firearm Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video source and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=7-hcS5hpMCM"
    url = "https://www.youtube.com/watch?v=PPPoEyLPGf4"
    url = "https://www.youtube.com/watch?v=gunNXngtOKo"
    url = "https://www.youtube.com/watch?v=j9q70YDQpX4"
    # Example usage:
    # main()  # Uses webcam by default
    # main(source='youtube', youtube_url=url, frame_skip=10)  # Process every 10th frame
    main(source='youtube', youtube_url=url, frame_skip=10)

