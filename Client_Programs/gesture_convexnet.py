import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import json
from model import LayerScale, StochasticDepth #for convexnet
from rembg import remove
from PIL import Image
import io
import os
class_names = ["Brightness_Decrease", "Brightness_Increase", "Chrome_Open", "Cursor_Movement", "Double_Click", "Initiation", "Left_Click", "Neutral", "PowerPoint_Open", "Right_Click", "Screenshot", "Scroll", "Shutdown", "Volume_Decrease", "Volume_Increase"]

# for convexnet
# Use the LayerScale and StochasticDepth classes that we copied from keras official code
best_model = load_model(r'E:\MajorProject\Gesture based HCI\GBHCI\Non_Git\Models\ConvNeXt-XL.h5', compile=False, custom_objects={"LayerScale": LayerScale, "StochasticDepth": StochasticDepth})

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

img_height = 384
img_width= 384 #for convexnet

# For webcam input:
cap = cv2.VideoCapture(0)

# Initialize frame counter
frame_counter = 0
# Set the number of frames to skip
skip_frames = 10  # Change this value to process more or fewer frames
img_count=1
bbox = None
predicted_class = None

# Create a directory if it doesn't exist
temp_input_dir_name = 'tmp'
if not os.path.exists(temp_input_dir_name):
    os.makedirs(temp_input_dir_name)
temp_output_dir_name = 'output'
if not os.path.exists(temp_output_dir_name):
    os.makedirs(temp_output_dir_name)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Only process every nth frame
    if frame_counter % skip_frames == 0:
        # Convert the BGR image to RGB and process it with MediaPipe Hands
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw bounding box around the hand with some padding
                hand_landmarks_array = np.array([[data.x, data.y, data.z] for data in hand_landmarks.landmark])
                x_min, y_min, z_min = np.min(hand_landmarks_array, axis=0)
                x_max, y_max, z_max = np.max(hand_landmarks_array, axis=0)
                padding = 0.05  # Change this value to increase/decrease the padding
                x_min -= padding
                y_min -= padding
                x_max += padding
                y_max += padding
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(1, x_max)
                y_max = min(1, y_max)
                bbox = np.array([x_min * frame.shape[1], y_min * frame.shape[0], x_max * frame.shape[1], y_max * frame.shape[0]]).astype(int)


                # Extract the hand image
                hand_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            

                # Preprocess the hand image
                img = cv2.resize(hand_img, (img_height, img_width))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                predictions = best_model.predict(img)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])

                # Print the class name and confidence
                print('The predicted class is:', class_names[predicted_class])
                print('Confidence:', confidence)

    # Draw the bounding box
    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)  # Change the last parameter to adjust the thickness of the bounding box

    # Display the class name above the bounding box
    if predicted_class is not None:
        cv2.putText(frame, class_names[predicted_class], (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Increment the frame counter
    frame_counter += 1

    # Display the resulting frame
    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

os.rmdir(temp_input_dir_name)
os.rmdir(temp_output_dir_name)