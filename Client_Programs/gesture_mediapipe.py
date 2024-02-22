# def run_mediapipe():
#     import cv2  
#     import mediapipe as mp  
#     import pyautogui 

#     # Start the webcam video capture
#     cap = cv2.VideoCapture(0)

#     # Initialize the hand detection model from MediaPipe
#     hand_detector = mp.solutions.hands.Hands(max_num_hands=1)

#     # Get the size of the screen
#     screen_width, screen_height = pyautogui.size()

#     # Initialize index_y
#     index_y = 0

#     # Main loop
#     while True:
#         # Capture a frame from the webcam
#         _, frame = cap.read()

#         # Flip the frame
#         frame = cv2.flip(frame, 1)

#         # Get the height and width of the frame
#         frame_height, frame_width, _ = frame.shape

#         # Convert the frame to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Pass the frame to the hand detector
#         output = hand_detector.process(rgb_frame)

#         # Get the hand landmarks
#         hands = output.multi_hand_landmarks

#         # If any hands are detected
#         if hands:
#             # For each hand
#             for hand in hands:
#                 # Get the landmarks
#                 landmarks = hand.landmark

#                 # For each landmark
#                 for id, landmark in enumerate(landmarks):
#                     # Get the x and y coordinates of the landmark
#                     x = int(landmark.x*frame_width)
#                     y = int(landmark.y*frame_height)

#                     # If the landmark is the tip of the index finger
#                     if id == 8:
#                         # Draw a circle at the position of the landmark
#                         cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))

#                         # Calculate the corresponding position on the screen
#                         index_x = screen_width/frame_width*x
#                         index_y = screen_height/frame_height*y

#                     # If the landmark is the tip of the thumb
#                     if id == 4:
#                         # Draw a circle at the position of the landmark
#                         cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))

#                         # Calculate the corresponding position on the screen
#                         thumb_x = screen_width/frame_width*x
#                         thumb_y = screen_height/frame_height*y

#                         # If the vertical distance between the thumb and index finger is less than 20
#                         if abs(index_y - thumb_y) < 30:
#                             # Simulate a mouse click
#                             pyautogui.click()

#                             # Sleep for 1 second
#                             pyautogui.sleep(2)

#                         # If the vertical distance between the thumb and index finger is less than 100
#                         elif abs(index_y - thumb_y) < 100:
#                             # Move the mouse pointer to the position of the index finger
#                             pyautogui.moveTo(index_x, index_y)

#         # Display the frame with the hand landmarks
#         cv2.imshow('GBHCI', frame)

#         # If the 'Esc' key is pressed, break the loop
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

# # Protect the main part of the code
# if __name__ == "__main__":
#     run_mediapipe()
import cv2  
import mediapipe as mp  
import pyautogui 

def run_mediapipe():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
    screen_width, screen_height = pyautogui.size()

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        index_x = index_y = thumb_x = thumb_y = None  # Initialize variables to None

        if hands:
            for hand in hands:
                landmarks = hand.landmark
                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x*frame_width)
                    y = int(landmark.y*frame_height)

                    if id == 8:
                        cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                        index_x = screen_width/frame_width*x
                        index_y = screen_height/frame_height*y

                    if id == 4:
                        cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                        thumb_x = screen_width/frame_width*x
                        thumb_y = screen_height/frame_height*y

                if index_y is not None and thumb_y is not None:  # Check if variables are not None
                    if abs(index_y - thumb_y) < 30:
                        pyautogui.click()
                        pyautogui.sleep(2)

                    elif abs(index_y - thumb_y) < 100:
                        pyautogui.moveTo(index_x, index_y)

        cv2.imshow('GBHCI', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    run_mediapipe()
