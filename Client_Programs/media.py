#the code below is a version with less preprocessing of gesture.py

import cv2
import mediapipe as mp
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize MediaPipe DrawingUtils.
mp_drawing = mp.solutions.drawing_utils
img_height = 380
img_width= 380
# OpenCV VideoCapture.
cap = cv2.VideoCapture(0)



# Load the trained model for efficientnet
model = load_model('../Models/V2M_alpha.h5')

# Create the directory if it doesn't exist.
if not os.path.exists('test_a'):
    os.makedirs('test_a')

# Initialize a counter for the image filenames.
img_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the image horizontally for a later selfie-view display.
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and find hands.
    results = hands.process(rgb_frame)

    # Check if any hands are detected.
    if results.multi_hand_landmarks is not None:
        # Draw the hand annotations on the image.
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box coordinates.
            x_min = int(min(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
            y_min = int(min(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0])
            x_max = int(max(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
            y_max = int(max(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0])

            # Add padding to the bounding box coordinates.
            padding = 20  # Define the padding size.
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)

            # Draw the bounding box.
            # For the text background
            # Finds space required by the text so that we can put a background with that amount of width.
            (w, h), _ = cv2.getTextSize(
                    "label", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Prints the text.    
            img = cv2.rectangle(frame, (x_min, y_min - 20), (x_min + w, y_min), (0,0,0), -1)
            img = cv2.putText(frame, "label", (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

            # For printing text
            img = cv2.putText(frame, 'test', (x_min, y_min),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    
    # Display the resulting frame.
    cv2.imshow('MediaPipe Hands', frame)

    # Exit loop if 'q' is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close OpenCV windows.
cap.release()
cv2.destroyAllWindows()
