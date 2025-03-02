#!/usr/bin/env python3
#
# micro_expressions.py

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize variables for plotting
emotions = ['Happiness', 'Sadness', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt']
emotion_values = [0] * len(emotions)

# Set up the plot
fig, ax = plt.subplots()
bars = ax.bar(emotions, emotion_values)
ax.set_ylim(0, 1)
ax.set_ylabel('Confidence')
ax.set_title('Real-time Micro-expression Analysis')

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_angle(point1, point2, point3):
    vector1 = np.array([point1.x - point2.x, point1.y - point2.y])
    vector2 = np.array([point3.x - point2.x, point3.y - point2.y])
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def analyze_emotions(landmarks):
    # Base measurements for normalization
    face_width = calculate_distance(landmarks[234], landmarks[454])  # Outer corners of eyes
    face_height = calculate_distance(landmarks[10], landmarks[152])  # Chin to forehead

    # Happiness
    smile_ratio = calculate_distance(landmarks[61], landmarks[291]) / face_width
    eye_corner_ratio = (calculate_distance(landmarks[243], landmarks[33]) + calculate_distance(landmarks[463], landmarks[263])) / (2 * face_width)
    cheek_raise = (calculate_distance(landmarks[117], landmarks[50]) + calculate_distance(landmarks[346], landmarks[280])) / (2 * face_height)
    happiness = (smile_ratio * 0.5 + eye_corner_ratio * 0.3 + cheek_raise * 0.2) * 2  # Weighted sum, multiplied for emphasis

    # Sadness
    inner_brow_distance = calculate_distance(landmarks[52], landmarks[282]) / face_width
    mouth_corner_angle = (calculate_angle(landmarks[61], landmarks[291], landmarks[0]) + calculate_angle(landmarks[291], landmarks[61], landmarks[17])) / 2
    lip_corner_depression = (calculate_distance(landmarks[61], landmarks[14]) + calculate_distance(landmarks[291], landmarks[14])) / (2 * face_height)
    sadness = ((1 - inner_brow_distance) * 0.3 + (180 - mouth_corner_angle) / 180 * 0.4 + lip_corner_depression * 0.3) * 1.5

    # Surprise
    eye_aspect_ratio = (calculate_distance(landmarks[386], landmarks[374]) + calculate_distance(landmarks[159], landmarks[145])) / (2 * face_width)
    mouth_aspect_ratio = calculate_distance(landmarks[13], landmarks[14]) / face_width
    brow_raise = (calculate_distance(landmarks[52], landmarks[282]) + calculate_distance(landmarks[66], landmarks[296])) / (2 * face_width)
    surprise = (eye_aspect_ratio * 0.4 + mouth_aspect_ratio * 0.3 + brow_raise * 0.3) * 2

    # Fear
    eye_white_ratio = (calculate_distance(landmarks[386], landmarks[374]) + calculate_distance(landmarks[159], landmarks[145])) / (2 * calculate_distance(landmarks[33], landmarks[133]))
    brow_raise_center = calculate_distance(landmarks[10], landmarks[151]) / face_height
    lip_stretch = calculate_distance(landmarks[61], landmarks[291]) / face_width
    fear = (eye_white_ratio * 0.4 + brow_raise_center * 0.3 + lip_stretch * 0.3) * 1.5

    # Anger
    brow_lowerer = 1 - (calculate_distance(landmarks[52], landmarks[282]) / face_width)
    eye_aperture = 1 - ((calculate_distance(landmarks[386], landmarks[374]) + calculate_distance(landmarks[159], landmarks[145])) / (2 * face_width))
    lip_tightener = 1 - (calculate_distance(landmarks[61], landmarks[291]) / face_width)
    chin_raiser = calculate_distance(landmarks[17], landmarks[14]) / face_height
    anger = (brow_lowerer * 0.3 + eye_aperture * 0.2 + lip_tightener * 0.3 + chin_raiser * 0.2) * 1.5

    # Disgust
    nose_wrinkler = 1 - (calculate_distance(landmarks[168], landmarks[19]) / face_height)
    upper_lip_raiser = calculate_distance(landmarks[0], landmarks[17]) / calculate_distance(landmarks[61], landmarks[291])
    lower_lip_depressor = calculate_distance(landmarks[17], landmarks[14]) / face_height
    disgust = (nose_wrinkler * 0.4 + upper_lip_raiser * 0.4 + lower_lip_depressor * 0.2) * 1.5

    # Contempt
    mouth_corner_pull = abs(calculate_distance(landmarks[61], landmarks[0]) - calculate_distance(landmarks[291], landmarks[17])) / face_width
    lip_corner_tightener = 1 - (calculate_distance(landmarks[61], landmarks[291]) / face_width)
    cheek_raiser = (calculate_distance(landmarks[117], landmarks[50]) + calculate_distance(landmarks[346], landmarks[280])) / (2 * face_height)
    contempt = (mouth_corner_pull * 0.5 + lip_corner_tightener * 0.3 + cheek_raiser * 0.2) * 1.5

    # Normalize values
    emotions = [happiness, sadness, surprise, fear, anger, disgust, contempt]
    total = sum(emotions)
    normalized_emotions = [max(0, min(e/total, 1)) for e in emotions]  # Ensure values are between 0 and 1

    return normalized_emotions

def update_plot(frame):
    ret, image = cap.read()
    if not ret:
        return bars

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Face Mesh
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Analyze emotions based on landmark positions
        emotion_values = analyze_emotions(landmarks)

        # Update the bar heights
        for bar, value in zip(bars, emotion_values):
            bar.set_height(value)

    return bars

# Create the animation
anim = FuncAnimation(fig, update_plot, frames=200, interval=50, blit=True)

plt.show()

# Release the camera when done
cap.release()
