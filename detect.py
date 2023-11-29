import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('GBHCI_5epoch.h5')

# Open the webcam (change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Define the input image size expected by the model
input_size = (224, 224)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Resize the frame to a larger size
    enlarged_frame = cv2.resize(frame, (640, 480))  # Example larger size
    
    # Calculate the center crop coordinates
    y_center = enlarged_frame.shape[0] // 2
    x_center = enlarged_frame.shape[1] // 2
    half_size = input_size[0] // 2
    
    # Extract the center region (ROI)
    roi = enlarged_frame[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]
    
    # Expand dimensions to match model input shape (batch size of 1)
    input_frame = np.expand_dims(roi, axis=0)
    
    # Make a prediction
    prediction = model.predict(input_frame)
    
    # Get the class label with the highest probability
    predicted_class = np.argmax(prediction)
    
    # Draw bounding box around the gesture (optional)
    cv2.rectangle(enlarged_frame, (x_center - half_size, y_center - half_size), (x_center + half_size, y_center + half_size), (0, 0, 255), 2)
    cv2.putText(enlarged_frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the enlarged frame with bounding box and predicted class
    cv2.imshow('Gesture Prediction', enlarged_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()


# #below predicts on images 

# import tensorflow as tf
# from keras.preprocessing import image
# from efficientnet.keras import EfficientNetB4
# import numpy as np

# # Load the trained model
# model_path = 'GBHCI_EfficientNetB4_a.h5'
# loaded_model = tf.keras.models.load_model(model_path)

# # Load and preprocess an image for prediction
# image_path = 'Dataset_alpha_split/validation/Nothing/image_7.jpg'  # Change this to the path of your image
# img = image.load_img(image_path, target_size=(380, 380))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0

# # Make a prediction
# predictions = loaded_model.predict(img_array)
# class_index = np.argmax(predictions[0])
# class_label = loaded_model.class_labels[class_index]

# print("Predicted class:", class_label)
# print("Predicted probabilities:", predictions[0])

# import cv2
# import mediapipe as mp
# import numpy as np
# from keras.models import load_model
# import tensorflow as tf

# # Define the custom layer (FixedDropout)
# class FixedDropout(tf.keras.layers.Layer):
#     def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
#         super(FixedDropout, self).__init__(**kwargs)
#         self.rate = rate
#         self.noise_shape = noise_shape
#         self.seed = seed

#     def call(self, inputs, training=None):
#         return tf.keras.backend.dropout(inputs, level=self.rate, noise_shape=self.noise_shape, seed=self.seed)

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# # Load your gesture recognition model with custom objects
# custom_objects = {'FixedDropout': FixedDropout}
# gesture_model = tf.keras.models.load_model('GBHCI_EfficientNetB4_a.h5', custom_objects=custom_objects)

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Convert the BGR image to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Detect hands
#     results = hands.process(rgb_frame)

#     if results.multi_hand_landmarks:
#         for landmarks in results.multi_hand_landmarks:
#             bbox = []
#             for id, lm in enumerate(landmarks.landmark):
#                 h, w, _ = frame.shape
#                 x, y = int(lm.x * w), int(lm.y * h)
#                 bbox.append((x, y))
#                 cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            
#             # Calculate bounding box dimensions
#             x_min = min(bbox, key=lambda item: item[0])[0]
#             x_max = max(bbox, key=lambda item: item[0])[0]
#             y_min = min(bbox, key=lambda item: item[1])[1]
#             y_max = max(bbox, key=lambda item: item[1])[1]
            
#             # Extract hand region
#             hand_roi = frame[y_min:y_max, x_min:x_max]
            
#             # Preprocess hand image for your gesture recognition model
#             hand_resized = cv2.resize(hand_roi, (380, 380))
#             hand_normalized = hand_resized / 255.0
#             hand_final = np.expand_dims(hand_normalized, axis=0)
            
#             # Predict gesture using your model
#             gesture_probabilities = gesture_model.predict(hand_final)
#             predicted_gesture = np.argmax(gesture_probabilities)
            
#             # Display the predicted gesture on the frame
#             gesture_text = f'Gesture: {predicted_gesture}'
#             cv2.putText(frame, gesture_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
#             # Draw bounding box
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#     cv2.imshow('Hand Gesture Recognition', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

