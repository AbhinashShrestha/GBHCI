import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import json
from rembg import remove
from PIL import Image
import io
import os
from actions import *
import time
# Initialize a variable to hold the time of the last detection
last_detection_time = None

skip_frames = 5
# Load class_names
#Keras’s image_dataset_from_directory function generates labels as integer indices that 
# correspond to the alphabetical order of the class names. 
# when we use np.argmax(predictions[0]) to get the predicted class index,
# this index should correspond to the correct class name in the class_names list
# with open('../class/class_names.json', 'r') as f:
class_names = ["Brightness_Decrease", "Brightness_Increase", "Chrome_Open", "Cursor_Movement", "Double_Click", "Initiation", "Left_Click", "Neutral", "PowerPoint_Open", "Right_Click", "Screenshot", "Scroll", "Shutdown", "Volume_Decrease", "Volume_Increase"]
    
# print(class_names)
# Load the trained model for efficientnet
model = load_model(r'E:\MajorProject\Gesture based HCI\GBHCI\Non_Git\Models\Resnet50v2_FEB_14_Dataset_alpha_finetuned.h5')


# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)


img_height = 224
img_width= 224

# For webcam input:
cap = cv2.VideoCapture(0)

# Initialize frame counter
frame_counter = 0
img_count=1
bbox = None
predicted_class = None

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
                #the below will draw landmarks but if that is given to the model then the model will predict all classes to be nothing
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                
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
                
                # With background remove Preprocess the hand image
                img = cv2.resize(hand_img, (img_height, img_width))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                
                #below is rembg preprocessing

                # Convert the OpenCV image (numpy.ndarray) to PIL.Image
                # input_hand = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

                # # Remove the background
                # output = remove(np.array(input_hand))

                # # Convert the result back to numpy.ndarray for cv2 functions
                # bg_removed_hand_img = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        
                # # The cv2.resize function expects an image read by OpenCV (which is a NumPy array)
                # img = cv2.resize(bg_removed_hand_img, (img_height, img_width))
                # img = image.img_to_array(img)
                # img = np.expand_dims(img, axis=0)

                # Use the model to predict the class
                predictions = model.predict(img) 
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                # handler = ActionHandler(class_names[predicted_class])
                # handler.execute_action()

                # Print the class name and confidence
                print('The predicted class is:', class_names[predicted_class])
                print('Confidence:', confidence)

    # # Draw the bounding box
    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 3)  # Change the last parameter to adjust the thickness of the bounding box

    # Display the class name above the bounding box
    if predicted_class is not None:
        cv2.putText(frame, class_names[predicted_class],(bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), thickness=2,lineType=cv2.LINE_AA)    
    
    # Increment the frame counter
    frame_counter += 1

    # Display the resulting frame
    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()