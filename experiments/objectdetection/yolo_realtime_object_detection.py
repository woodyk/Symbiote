#!/usr/bin/env python3
#
# yolo_tracking.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
from yt_dlp import YoutubeDL
from ultralytics import YOLO
import logging
import json
import time

# Suppress ultralytics model logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

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

def main(source='webcam', youtube_url=None):
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Initialize the video source
    cap = get_video_source(source, youtube_url)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

    # Dictionary to keep track of active objects and their first detection time
    active_objects = {}

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_objects = set()

        # Perform object detection
        results = model(frame)

        # Loop over the detections
        for result in results:
            boxes = result.boxes  # Boxes object for the frame
            for box in boxes:
                # Extract the bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                # Get the class name
                object_name = model.names[cls]
                current_objects.add(object_name)

                if object_name not in active_objects:
                    # First time seeing this object, log it with the timestamp
                    timestamp = get_timestamp(frame_number, fps)
                    active_objects[object_name] = {"appeared_at": timestamp}

                    # Create a dictionary with detection details
                    detection_info = {
                        "event": "appeared",
                        "object_name": object_name,
                        "confidence": round(float(conf), 2),
                        "coordinates": {
                            "x_min": int(xyxy[0]),
                            "y_min": int(xyxy[1]),
                            "x_max": int(xyxy[2]),
                            "y_max": int(xyxy[3])
                        },
                        "width": int(xyxy[2] - xyxy[0]),
                        "height": int(xyxy[3] - xyxy[1]),
                        "timestamp": timestamp
                    }

                    # Print the dictionary as a JSON string
                    print(json.dumps(detection_info, indent=4))

                # Draw the bounding box
                label = f'{object_name} {conf:.2f}'
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

                # Put the label above the bounding box
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check for disappeared objects
        disappeared_objects = set(active_objects.keys()) - current_objects
        for object_name in disappeared_objects:
            # Log the disappearance with the timestamp
            timestamp = get_timestamp(frame_number, fps)
            disappearance_info = {
                "event": "disappeared",
                "object_name": object_name,
                "disappeared_at": timestamp
            }

            # Print the disappearance event as a JSON string
            print(json.dumps(disappearance_info, indent=4))

            # Remove the object from active_objects
            del active_objects[object_name]

        # Display the frame with bounding boxes
        cv2.imshow('YOLOv8 Object Tracking', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video source and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=7-hcS5hpMCM"
    # Example usage:
    # main()  # Uses webcam by default
    # main(source='youtube', youtube_url='https://www.youtube.com/watch?v=example')  # For YouTube video

    # Modify the following line to switch between webcam and YouTube video
    #main(source='youtube', youtube_url='https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    main(source='youtube', youtube_url=url)

