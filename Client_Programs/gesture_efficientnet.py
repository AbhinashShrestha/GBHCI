import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import json
# from rembg import remove
from PIL import Image
import io
import os
from actions import *
import time
# Initialize a variable to hold the time of the last detection
last_detection_time = None
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

skip_frames = 10
# Load class_names
#Keras’s image_dataset_from_directory function generates labels as integer indices that 
# correspond to the alphabetical order of the class names. 
# when we use np.argmax(predictions[0]) to get the predicted class index,
# this index should correspond to the correct class name in the class_names list
# class_names = ['Brightness_Decrease', 'Brightness_Increase', 'Chrome_Open', 'Cursor_Movement', 'Double_Click', 'Left_Click', 'Nothing', 'PowerPoint_Open', 'Right_Click', 'Screenshot', 'Scroll', 'Shutdown', 'VSCode_Open', 'Volume_Decrease', 'Volume_Increase']
#for asl mix
# class_names= ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

#for asl with no bg
# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#for final asl_mix 
# class_names = ['A', 'B', 'Blank', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'U', 'V', 'W', 'X', 'Y']
# class_names = ['A', 'B', 'Blank', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'U', 'V', 'W', 'X', 'Y']


#final feb 20
# class_names = ['5', 'A', 'B', 'Blank', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'V', 'W', 'X', 'Y', 'space']


#final feb 21
# class_names = ['3', '5', 'A', 'B', 'Blank', 'C', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'V', 'X', 'Y', 'space']

#final feb 21 there are 19 classes
# class_names = ['A', 'B', 'Blank', 'C', 'E', 'F', 'G', 'H', 'K', 'L', 'N', 'O', 'P', 'Q', 'S', 'V', 'X', 'Y', 'space']

#feb 22 Final there are 18 classes
class_names = ['A', 'B', 'Blank', 'C', 'E', 'F', 'G', 'H', 'L', 'N', 'O', 'P', 'Q', 'S', 'V', 'X', 'Y', 'space']

# Define the mapping between class names and labels
class_label_mapping = {
    'A': 'Shutdown',
    'B': 'Scroll_Up',
    'Blank': 'Anomaly',
    'C': 'Chrome_Open',
    'E': 'Screenshot',
    'F': 'Scroll_Down',
    'G': 'Double_Click',
    'H': 'Right_Click',
    # 'K': 'VSCode_Open',
    'L': 'Left_Click',
    'N': 'N',
    'O': 'PowerPoint_Open',
    'P': 'Brightness_Increase',
    'Q': 'Brightness_Decrease',
    'S': 'Neutral',
    'V': 'VSCode_Open',
    'X': 'Volume_Decrease',
    'Y': 'Volume_Increase',
    'space':'Restart'
}

# Create a dictionary that maps indices to labels
class_labels = {i: class_label_mapping[name] for i, name in enumerate(class_names)}


model = load_model(r'E:\MajorProject\Gesture based HCI\GBHCI\Non_Git\Models\EV2B2\EfficientNetV2B2_FEB_22_batch_size_64_18_classes_A_B_Blank_C_E_F_G_H_L_N_O_P_Q_S_V_X_Y_space_finetuned.h5')

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

#depends of the efficientnet architecture
img_height = 300 
img_width = 300
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

                # Use the model to predict the class
                predictions = model.predict(img) 
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                handler = ActionHandler(class_labels[predicted_class])
                handler.execute_action()

                # Print the class name and confidence
                print('The predicted class is:', class_labels[predicted_class])
                print('Confidence:', confidence)

    # # Draw the bounding box
    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 3)  # Change the last parameter to adjust the thickness of the bounding box

    # Display the class name above the bounding box
    if predicted_class is not None:
        cv2.putText(frame, class_labels[predicted_class],(bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), thickness=2,lineType=cv2.LINE_AA)    
    
    # Increment the frame counter
    frame_counter += 1

    # Display the resulting frame
    cv2.imshow('GBHCI', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()