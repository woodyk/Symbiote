#!/usr/bin/env python3

import random
import cv2
import numpy as np

# initialize the frame
frame = np.zeros((height, width))

# main loop
while True:
    # generate static
    for i in range(height):
        for j in range(width):
            frame[i][j] = random.choice([0, 255])

    # find contours
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    for contour in contours:
        cv2.drawContours(frame, contour, -1, (255, 255, 255), 3)

    # apply decay
    frame = cv2.addWeighted(frame, 0.9, np.zeros_like(frame), 0.1, 0)

    # display frame
    cv2.imshow('Whisping Smoke', frame)

    # check for spacebar press to pause
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.waitKey(-1)

    # check for 'q' press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

