import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize MediaPipe DrawingUtils.
mp_drawing = mp.solutions.drawing_utils

# OpenCV VideoCapture.
cap = cv2.VideoCapture(0)

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

    # Draw the hand annotations on the image.
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get the bounding box coordinates.
        x_min = int(min(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
        y_min = int(min(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0])
        x_max = int(max(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
        y_max = int(max(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0])

        # Draw the bounding box.
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Capture the image within the bounding box.
        captured_image = frame[y_min:y_max, x_min:x_max]

    # Display the resulting frame.
    cv2.imshow('MediaPipe Hands', frame)

    # Exit loop if 'q' is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close OpenCV windows.
cap.release()
cv2.destroyAllWindows()
