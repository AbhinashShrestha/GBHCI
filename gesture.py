# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# # Assuming your data directory is organized with one subdirectory per class
# data_dir = "Dataset_alpha"
# class_names = sorted(os.listdir(data_dir))


# # Load the trained model
# model = load_model('Models/V2M_alpha.h5')
# img_height = 380
# img_width=380
# # Load the image
# img_path = 'classmates_test_images/aa.png'  # replace with the path to your image
# img = image.load_img(img_path, target_size=(img_height, img_width))

# # Preprocess the image
# img_array = image.img_to_array(img)
# img_batch = np.expand_dims(img_array, axis=0)

# # Use the model to predict the class
# predictions = model.predict(img_batch)

# # The predictions are softmax probabilities, to get the class we find the index of the highest probability
# predicted_class = np.argmax(predictions[0])
# confidence = np.max(predictions[0])

# # Print the class name and confidence
# print('The predicted class is:', class_names[predicted_class])
# print('Confidence:', confidence)

# real working
import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import time
import json

# Load class_names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
    
print(class_names)
# # Load the trained model
# model = load_model('Models/V2M_alpha.h5')

# # Initialize MediaPipe Hands
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# img_height = 380
# img_width=380

# # For webcam input:
# cap = cv2.VideoCapture(0)

# # Initialize time for delay
# t0 = time.time()

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     # Check if 10 seconds have passed
#     if time.time() - t0 >= 10:
#         # Convert the BGR image to RGB and process it with MediaPipe Hands
#         results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # Draw the hand annotations on the image
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw bounding box around the hand with some padding
#                 hand_landmarks_array = np.array([[data.x, data.y, data.z] for data in hand_landmarks.landmark])
#                 x_min, y_min, z_min = np.min(hand_landmarks_array, axis=0)
#                 x_max, y_max, z_max = np.max(hand_landmarks_array, axis=0)
#                 padding = 0.05  # Change this value to increase/decrease the padding
#                 x_min -= padding
#                 y_min -= padding
#                 x_max += padding
#                 y_max += padding
#                 x_min = max(0, x_min)
#                 y_min = max(0, y_min)
#                 x_max = min(1, x_max)
#                 y_max = min(1, y_max)
#                 bbox = np.array([x_min * frame.shape[1], y_min * frame.shape[0], x_max * frame.shape[1], y_max * frame.shape[0]]).astype(int)

#                 # Extract the hand image
#                 hand_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

#                 # Preprocess the hand image
#                 img = cv2.resize(hand_img, (img_height, img_width))
#                 img = image.img_to_array(img)
#                 img = np.expand_dims(img, axis=0)

#                 # Use the model to predict the class
#                 predictions = model.predict(img)
#                 predicted_class = np.argmax(predictions[0])
#                 confidence = np.max(predictions[0])

#                 # Print the class name and confidence
#                 print('The predicted class is:', class_names[predicted_class])
#                 print('Confidence:', confidence)

#                 # Draw the bounding box
#                 cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

#         # Reset the time
#         t0 = time.time()

#     # Display the resulting frame
#     cv2.imshow('MediaPipe Hands', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
