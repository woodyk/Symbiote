#!/usr/bin/env python3
#
# microexpression.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from deepface import DeepFace
import time

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize variables for plotting
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
races = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
genders = ['Man', 'Woman']

emotion_values = [0] * len(emotions)
race_values = [0] * len(races)
gender_values = [0] * len(genders)

# Set up the plot
fig, axs = plt.subplots(5, 1, figsize=(10, 25))

# Emotion bar plot
bars_emotion = axs[0].bar(emotions, emotion_values)
axs[0].set_ylim(0, 100)
axs[0].set_ylabel('Confidence (%)')
axs[0].set_title('Real-time Emotion Analysis')

# Age plot
age_line, = axs[1].plot([], [], 'r-')
axs[1].set_xlim(0, 200)
axs[1].set_ylim(0, 100)
axs[1].set_ylabel('Age')
axs[1].set_title('Real-time Age Estimation')
age_data = []

# Gender bar plot
bars_gender = axs[2].bar(genders, gender_values)
axs[2].set_ylim(0, 100)
axs[2].set_ylabel('Confidence (%)')
axs[2].set_title('Real-time Gender Analysis')

# Race bar plot
bars_race = axs[3].bar(races, race_values)
axs[3].set_ylim(0, 100)
axs[3].set_ylabel('Confidence (%)')
axs[3].set_title('Real-time Race/Ethnicity Analysis')

# For displaying the video feed
img_plot = axs[4].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
axs[4].axis('off')

last_analysis_time = 0
analysis_interval = 0.1  # Analyze every 100ms

def update_plot(frame):
    global last_analysis_time, age_data
    ret, image = cap.read()
    if not ret:
        return tuple(bars_emotion) + tuple(bars_gender) + tuple(bars_race) + (age_line, img_plot)

    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        try:
            # Analyze face
            result = DeepFace.analyze(image,
                                      actions=['emotion', 'age', 'gender', 'race'],
                                      enforce_detection=False,
                                      detector_backend='mtcnn',
                                      silent=True)

            # Print the result to understand its structure and values
            print(result)

            if result:
                # Access the first detected face's data
                face_data = result[0]  # Result is a list, so we access the first element

                # Update emotion bars
                emotion_preds = face_data['emotion']
                for i, emotion in enumerate(emotions):
                    bars_emotion[i].set_height(emotion_preds[emotion])

                # Update age line
                age = face_data['age']
                age_data.append(age)
                age_line.set_data(range(len(age_data)), age_data)
                axs[1].set_xlim(0, max(200, len(age_data)))

                # Update gender bars
                gender_preds = face_data['gender']
                for i, gender in enumerate(genders):
                    bars_gender[i].set_height(gender_preds[gender])

                # Update race bars
                race_preds = face_data['race']
                for i, race in enumerate(races):
                    bars_race[i].set_height(race_preds[race])

        except Exception as e:
            print(f"An error occurred: {e}")

        last_analysis_time = current_time

    # Update the video feed
    img_plot.set_array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return tuple(bars_emotion) + tuple(bars_gender) + tuple(bars_race) + (age_line, img_plot)

# Create the animation
anim = FuncAnimation(fig, update_plot, frames=200, interval=50, blit=True)

plt.tight_layout()
plt.show()

# Release the camera when done
cap.release()

