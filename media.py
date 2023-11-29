import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize a dictionary to store the last saved hand landmarks
last_saved_landmarks = {}

# Initialize the image number
img_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            bbox = []
            landmark_points = []
            
            for id, lm in enumerate(landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                bbox.append((x, y))
                landmark_points.append((x, y))
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            
            # Calculate bounding box dimensions
            x_min = min(bbox, key=lambda item: item[0])[0]
            x_max = max(bbox, key=lambda item: item[0])[0]
            y_min = min(bbox, key=lambda item: item[1])[1]
            y_max = max(bbox, key=lambda item: item[1])[1]
            
            # Expand the bounding box dimensions by a percentage
            expand_percentage = 0.1
            expand_x = int((x_max - x_min) * expand_percentage)
            expand_y = int((y_max - y_min) * expand_percentage)
            x_min = max(0, x_min - expand_x)
            x_max = min(frame.shape[1], x_max + expand_x)
            y_min = max(0, y_min - expand_y)
            y_max = min(frame.shape[0], y_max + expand_y)
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw lines between hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # If the current hand landmarks are different from the last saved ones, save the image
            if str(landmark_points) != last_saved_landmarks.get('landmarks'):
                cv2.imwrite(f'{img_num}.png', frame)
                last_saved_landmarks['landmarks'] = str(landmark_points)
                img_num += 1

    cv2.imshow('Hand Detection with Bounding Box and Bones', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
